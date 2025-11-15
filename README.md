Medical Literature Extraction Pipeline V2 (Config-First)

Quick start

- `source venv/bin/activate` (or your environment)
- Install deps: `pip install -r requirements.txt`
- Create a `.env` with any provider secrets (e.g., `OPENAI_API_KEY`)

Run commands (current CLI)

- `python -m src.cli validate --config config/pipelines/doi.yaml`
- `python -m src.cli run --config config/pipelines/doi.yaml --input data/final_test.csv --id-column DOI --limit 5`
- `python -m src.cli test --config config/pipelines/doi.yaml --input data/final_test.csv`
- `python -m src.cli retry --config config/pipelines/doi.yaml --session <session-id>`

Alternative entrypoint

- `python cli.py run --config config/pipelines/doi.yaml ...` also works

Config selection shortcuts

- If `--config` is omitted, an interactive selector lists configs found under `config/pipelines/`
- You can set `MED_CONFIG_PATH` to point to a default config (skips interactive selection)

Notes

- The engine loads YAML overlays with recursive `include:` and performs `${VAR}` env var substitution.
- Prompts can be provided under `prompts.system` and `prompts.extraction`, or via included prompt YAMLs with top-level `system` and `extraction` keys.
- Outputs are written under `output/extracted/sessions/<pipeline>/<session_id>/records/` with a session `results.csv`. Audit logs and summaries are under `output/extracted/_audit/`.

Commands present in this trimmed CLI: `test`, `run`, `validate`, `retry`.
Previously documented commands like `preview`, `benchmark`, `export`, and the AI generator are not part of the current CLI build.
