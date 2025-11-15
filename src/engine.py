from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config.schema import ConfigSchema
from .models import ExtractionResult, Record, TransparencyMetadata
from .services.audit_service import AuditService
from .services.input_handler import InputHandler
from .services.llm_service import LLMService
from .services.output_handler import OutputHandler
from .services.parser import Parser
from .services.prompt_builder import PromptBuilder
from .services.quality_service import QualityService
from .utils import ensure_dir


class ExtractionEngine:
    def __init__(self, config: ConfigSchema, audit_service: AuditService) -> None:
        self.cfg = config
        self.audit = audit_service

        # Extract input signals from quality config for QualityService
        input_signals = []
        if self.cfg.quality and self.cfg.quality.input_signals:
            input_signals = [s.model_dump() for s in self.cfg.quality.input_signals]

        self.quality = QualityService(
            min_confidence_threshold=self.cfg.quality.min_confidence_threshold,
            review_threshold=self.cfg.quality.review_threshold,
            input_signals=input_signals,
        )
        self.llm = LLMService(self.cfg)

        # Output handler with session-based structure
        out_dir = self.cfg.output.directory
        pipeline_name = self.cfg.pipeline or "extraction"
        session_id = self.audit.session_id  # Get session ID from audit service
        self.output = OutputHandler(
            out_dir,
            session_id=session_id,
            pipeline_name=pipeline_name
        )

        # Parser defaults (can be enhanced via config fields later)
        self.parser = Parser(pointer_map={}, normalization={})

        # Prompt builder from config templates if present
        extras = getattr(self.cfg, "model_extra", {}) or {}
        sys_t = (self.cfg.prompts.system if self.cfg.prompts else extras.get("system")) or ""
        usr_t = (self.cfg.prompts.extraction if self.cfg.prompts else extras.get("extraction")) or ""
        # Option variables are expected to be included in config overlays under 'variables'
        variables = self._collect_prompt_variables(self.cfg)
        self.prompts = PromptBuilder(system_template=sys_t, user_template=usr_t, variables=variables)

    def _collect_prompt_variables(self, cfg: ConfigSchema) -> Dict[str, Any]:
        vars: Dict[str, Any] = {}
        # Extras (from includes like fields YAML) live in model_extra
        extras = getattr(cfg, "model_extra", {}) or {}
        def join_opts(key: str) -> Optional[str]:
            vals = extras.get(key)
            if isinstance(vals, list):
                return "\n  - " + "\n  - ".join(str(v) for v in vals)
            return None

        # Create friendly strings used by existing prompts
        sd = join_opts("study_design")
        if sd:
            vars["study_design_options"] = sd
        ss = join_opts("subspecialty_focus")
        if ss:
            vars["subspecialty_focus_options"] = ss
        pt = join_opts("priority_topic")
        if pt:
            vars["priority_topic_options"] = pt
        country = join_opts("country")
        if country:
            vars["country_options"] = country

        # Allow overlays to include a 'variables' map directly
        extras_vars = extras.get("variables")
        if isinstance(extras_vars, dict):
            vars.update(extras_vars)

        return vars

    async def process_record_async(
        self,
        record: Record,
        *,
        force: bool = False,
        strategy: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        # Dedupe check (respect session/records structure and sanitize filename)
        safe_key = str(record.key).replace("/", "_").replace("\\", "_")
        out_path = self.output.records_dir / f"{safe_key}.yaml"
        if out_path.exists() and not force:
            self.audit.log_event("skipped", key=record.key, reason="exists")
            self.audit.increment(skipped=1)
            return None

        system_prompt, user_prompt = self.prompts.build(record.data)

        # Hash prompts for transparency
        sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:10] if system_prompt else None
        usr_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:10] if user_prompt else None

        # Execute LLM
        try:
            resp = await self.llm.execute_request(
                system_prompt=system_prompt, user_prompt=user_prompt, record=record, strategy=strategy
            )
            self.audit.log_llm_interaction(
                key=record.key,
                provider=resp.provider,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=resp.cost,
                processing_time=resp.processing_time,
            )
        except Exception as e:
            self.audit.log_failure(key=record.key, error=str(e), failure_category="llm_error", retry_count=0)
            return None

        # Parse and normalize
        extracted, normalized = self.parser.parse_and_normalize(resp.content)

        # Quality
        conf = self.quality.calculate_confidence(resp.confidence, record)
        valid, errors = self.quality.validate_extraction(normalized or extracted, conf)
        # Log low-confidence signal for observability
        try:
            if conf.overall < self.cfg.quality.review_threshold:
                self.audit.log_event("low_confidence", key=record.key, confidence=conf.overall)
        except Exception:
            pass

        result = ExtractionResult(
            key=record.key,
            input=record.data,
            extracted=extracted,
            normalized=normalized or extracted,
            confidence=conf,
            valid=valid,
            errors=errors,
            transparency=TransparencyMetadata(
                provider=resp.provider,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=resp.cost,
                processing_time=resp.processing_time,
                system_prompt_hash=sys_hash,
                user_prompt_hash=usr_hash,
            ),
            raw_llm=resp.raw_response if isinstance(resp.raw_response, dict) else None,
        )

        self.output.write_record_output(result)
        self.output.update_csv_aggregate(result)
        self.audit.increment(succeeded=1, cost=result.transparency.cost or 0.0)
        self.audit.log_event("completed", key=record.key, valid=result.valid)
        return result

    async def run(
        self,
        *,
        input_path: Path | str,
        id_column: str = "id",
        skip: int = 0,
        limit: Optional[int] = None,
        force: bool = False,
        strategy: Optional[str] = None,
    ) -> None:
        # Get column mapping from config if available
        column_map = self.cfg.input.column_map if self.cfg.input else None
        handler = InputHandler(input_path, id_column=id_column, column_map=column_map, skip=skip, limit=limit)
        records = list(handler.iter_records())
        self.audit.increment(total=len(records))

        concurrency = max(1, int(self.cfg.processing.batch_size))
        delay = max(0.0, float(self.cfg.processing.delay_between_requests))

        # Log session start with input metadata for retry workflows
        self.audit.log_event(
            "session_start",
            total=len(records),
            input_path=str(input_path),
            id_column=id_column,
        )

        # Progress UI
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )
        overall_task = progress.add_task("Overall", total=len(records))
        review_task = progress.add_task("Low-confidence", total=len(records))
        failure_task = progress.add_task("Failures", total=len(records))

        async def worker(batch: List[Record]):
            return await asyncio.gather(
                *[self.process_record_async(r, force=force, strategy=strategy) for r in batch]
            )

        # Run in batches of `concurrency`
        reduced_so_far = 0
        failed_prev = 0
        with progress:
            for i in range(0, len(records), concurrency):
                batch = records[i : i + concurrency]
                results = await worker(batch)
                # Update progress counts
                progress.update(overall_task, advance=len(batch))
                # Count reduced-confidence results in this batch
                batch_reduced = 0
                for res in results:
                    if res is None:
                        continue
                    try:
                        if res.confidence and res.confidence.overall < self.cfg.quality.review_threshold:
                            batch_reduced += 1
                    except Exception:
                        pass
                if batch_reduced:
                    reduced_so_far += batch_reduced
                    progress.update(review_task, advance=batch_reduced)
                # Failures updated via audit service; compute delta
                failed_delta = max(0, self.audit.summary.failed - failed_prev)
                if failed_delta:
                    progress.update(failure_task, advance=failed_delta)
                    failed_prev = self.audit.summary.failed
                if delay:
                    await asyncio.sleep(delay)

        self.audit.finalize_session()
