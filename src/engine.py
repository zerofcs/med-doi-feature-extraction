from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config.schema import ConfigSchema
from .models import Confidence, ExtractionResult, Record, TransparencyMetadata
from .services.audit_service import AuditService
from .services.input_handler import InputHandler
from .services.llm_service import LLMService
from .services.output_handler import OutputHandler
from .services.parser import Parser
from .services.prompt_builder import PromptBuilder
from .services.quality_service import QualityService
from .utils import ensure_dir


SCREENING_TEMPLATE = """\
Title: {title}
Abstract: {abstract}

I am screening papers for a systematic review on {topic}.

Decide if the following paper should be included or excluded from our systematic review. Consider the title and abstract of the article when making your decision.

Exclude:
{criteria}

Be lenient. I prefer to include papers by mistake rather than excluding papers by mistake

Respond with 'INCLUDE' or 'EXCLUDE'\
"""


class ExtractionEngine:
    def __init__(self, config: ConfigSchema, audit_service: AuditService) -> None:
        self.cfg = config
        self.audit = audit_service
        self.task = config.task or "extract"
        self.llm = LLMService(self.cfg)

        # Output handler with session-based structure
        pipeline_name = self.cfg.pipeline or "pipeline"
        self.output = OutputHandler(
            self.cfg.output.directory,
            session_id=self.audit.session_id,
            pipeline_name=pipeline_name,
        )

        if self.task == "screen":
            self._setup_screening()
        else:
            self._setup_extraction()

    def _setup_screening(self) -> None:
        screen = self.cfg.screen
        if not screen or not screen.topic:
            raise ValueError("Screen config with 'topic' is required for screening tasks")
        self.screen_topic = screen.topic
        self.screen_criteria = "\n".join(f"- {c}" for c in screen.criteria)

    def _setup_extraction(self) -> None:
        input_signals = []
        if self.cfg.quality and self.cfg.quality.input_signals:
            input_signals = [s.model_dump() for s in self.cfg.quality.input_signals]

        self.quality = QualityService(
            min_confidence_threshold=self.cfg.quality.min_confidence_threshold,
            review_threshold=self.cfg.quality.review_threshold,
            input_signals=input_signals,
        )

        self.parser = Parser(pointer_map={}, normalization={})

        extras = getattr(self.cfg, "model_extra", {}) or {}
        sys_t = (self.cfg.prompts.system if self.cfg.prompts else extras.get("system")) or ""
        usr_t = (self.cfg.prompts.extraction if self.cfg.prompts else extras.get("extraction")) or ""
        variables = self._collect_prompt_variables(self.cfg)
        self.prompts = PromptBuilder(system_template=sys_t, user_template=usr_t, variables=variables)

    def _collect_prompt_variables(self, cfg: ConfigSchema) -> Dict[str, Any]:
        vars: Dict[str, Any] = {}
        extras = getattr(cfg, "model_extra", {}) or {}

        def join_opts(key: str) -> Optional[str]:
            vals = extras.get(key)
            if isinstance(vals, list):
                return "\n  - " + "\n  - ".join(str(v) for v in vals)
            return None

        for key in ("study_design", "subspecialty_focus", "priority_topic", "country"):
            val = join_opts(key)
            if val:
                vars[f"{key}_options"] = val

        extras_vars = extras.get("variables")
        if isinstance(extras_vars, dict):
            vars.update(extras_vars)
        return vars

    def _build_screening_prompt(self, record: Record) -> str:
        title = str(record.data.get("Title") or "")
        abstract = str(record.data.get("Abstract") or "")
        return SCREENING_TEMPLATE.format(
            topic=self.screen_topic,
            criteria=self.screen_criteria,
            title=title,
            abstract=abstract,
        )

    @staticmethod
    def _parse_screening_response(text: str) -> str:
        upper = text.strip().upper()
        if "INCLUDE" in upper:
            return "INCLUDE"
        if "EXCLUDE" in upper:
            return "EXCLUDE"
        return "UNCLEAR"

    async def process_record_async(
        self,
        record: Record,
        *,
        force: bool = False,
        strategy: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        safe_key = str(record.key).replace("/", "_").replace("\\", "_")
        out_path = self.output.records_dir / f"{safe_key}.yaml"
        if out_path.exists() and not force:
            self.audit.log_event("skipped", key=record.key, reason="exists")
            self.audit.increment(skipped=1)
            return None

        if self.task == "screen":
            return await self._screen_record(record, strategy=strategy)
        else:
            return await self._extract_record(record, force=force, strategy=strategy)

    async def _screen_record(
        self, record: Record, *, strategy: Optional[str] = None
    ) -> Optional[ExtractionResult]:
        user_prompt = self._build_screening_prompt(record)

        try:
            resp = await self.llm.execute_request(
                system_prompt="", user_prompt=user_prompt, record=record,
                strategy=strategy, max_tokens=500,
            )
            self.audit.log_llm_interaction(
                key=record.key, provider=resp.provider, model=resp.model,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                cost=resp.cost, processing_time=resp.processing_time,
            )
        except Exception as e:
            self.audit.log_failure(key=record.key, error=str(e), failure_category="llm_error", retry_count=0)
            return None

        decision = self._parse_screening_response(resp.content)

        result = ExtractionResult(
            key=record.key,
            input={"Title": record.data.get("Title", ""), "Abstract": str(record.data.get("Abstract", ""))[:200]},
            extracted={"decision": decision},
            normalized={"decision": decision},
            confidence=Confidence(overall=1.0),
            valid=decision != "UNCLEAR",
            errors=["unclear_response"] if decision == "UNCLEAR" else [],
            transparency=TransparencyMetadata(
                provider=resp.provider, model=resp.model,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                cost=resp.cost, processing_time=resp.processing_time,
            ),
        )

        self.output.write_record_output(result)
        self.output.update_csv_aggregate(result)
        self.audit.increment(succeeded=1, cost=result.transparency.cost or 0.0)
        self.audit.log_event("completed", key=record.key, decision=decision)
        return result

    async def _extract_record(
        self, record: Record, *, force: bool = False, strategy: Optional[str] = None
    ) -> Optional[ExtractionResult]:
        system_prompt, user_prompt = self.prompts.build(record.data)
        sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:10] if system_prompt else None
        usr_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:10] if user_prompt else None

        try:
            resp = await self.llm.execute_request(
                system_prompt=system_prompt, user_prompt=user_prompt, record=record, strategy=strategy
            )
            self.audit.log_llm_interaction(
                key=record.key, provider=resp.provider, model=resp.model,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                cost=resp.cost, processing_time=resp.processing_time,
            )
        except Exception as e:
            self.audit.log_failure(key=record.key, error=str(e), failure_category="llm_error", retry_count=0)
            return None

        extracted, normalized = self.parser.parse_and_normalize(resp.content)
        conf = self.quality.calculate_confidence(resp.confidence, record)
        valid, errors = self.quality.validate_extraction(normalized or extracted, conf)

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
                provider=resp.provider, model=resp.model,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                cost=resp.cost, processing_time=resp.processing_time,
                system_prompt_hash=sys_hash, user_prompt_hash=usr_hash,
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
        column_map = self.cfg.input.column_map if self.cfg.input else None
        handler = InputHandler(input_path, id_column=id_column, column_map=column_map, skip=skip, limit=limit)
        records = list(handler.iter_records())
        self.audit.increment(total=len(records))

        concurrency = max(1, int(self.cfg.processing.batch_size))
        delay = max(0.0, float(self.cfg.processing.delay_between_requests))

        self.audit.log_event(
            "session_start",
            task=self.task,
            total=len(records),
            input_path=str(input_path),
            id_column=id_column,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )
        overall_task = progress.add_task("Processing", total=len(records))
        failure_task = progress.add_task("Failures", total=len(records))

        async def worker(batch: List[Record]):
            return await asyncio.gather(
                *[self.process_record_async(r, force=force, strategy=strategy) for r in batch]
            )

        failed_prev = 0
        with progress:
            for i in range(0, len(records), concurrency):
                batch = records[i : i + concurrency]
                await worker(batch)
                progress.update(overall_task, advance=len(batch))
                failed_delta = max(0, self.audit.summary.failed - failed_prev)
                if failed_delta:
                    progress.update(failure_task, advance=failed_delta)
                    failed_prev = self.audit.summary.failed
                if delay:
                    await asyncio.sleep(delay)

        self.audit.finalize_session()
