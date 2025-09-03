# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` — core modules (`extractor.py`, `models.py`, `audit.py`, `quality.py`, `utils.py`).
- Providers: `src/providers/` — LLM backends (`openai_provider.py`, `ollama_provider.py`).
- CLI: `cli.py` (entrypoint) delegates to `src/cli.py` (Typer app).
- Config: `config/settings.yaml` (providers, cost, batching) and `config/prompts.yaml` (system/user prompts).
- Data/Outputs: input Excel (e.g., `data-source.xlsx`, ignored); results in `output/` (`extracted/`, `failures/`, `logs/`).
- Env: `.env.example` for local secrets; `.env` is git-ignored.

## Build, Test, and Development Commands
- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Help: `python cli.py --help` and `python cli.py extract --help`.
- Preview data: `python cli.py preview -f data-source.xlsx -s 0 -r 10`.
- Single record test: `python cli.py test -f data-source.xlsx -p ollama`.
- Run extraction: `python cli.py extract --skip 0 --limit 100 --provider openai`.
- Retry failures: `python cli.py retry`.
- Export results: `python cli.py export --format excel`.
- Quality report: `python cli.py validate`.

## Coding Style & Naming Conventions
- Language: Python 3.9+. Use 4‑space indentation, type hints, and docstrings.
- Models: Use Pydantic models/enums in `src/models.py` for types and validation.
- Naming: snake_case for files/functions, PascalCase for classes, UPPER_SNAKE for constants.
- Config: YAML keys lower_snake; keep defaults in `config/settings.yaml`.
- CLI UX: Use Rich for console output in CLI only; library code should return values and avoid side effects.

## Testing Guidelines
- Framework: pytest (recommended). Place tests in `tests/` named `test_*.py`.
- Run: `pytest -q` (install with `pip install pytest` for local dev).
- Focus: unit tests for `extractor`, providers, `quality`, and `audit`. Prefer small fixture rows over full spreadsheets.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`). Examples in history: `fix(cli): ...`, `docs: ...`.
- PRs: include purpose, linked issues, before/after behavior, CLI logs or screenshots, and any config changes. Update README if flags/flows change.

## Security & Configuration Tips
- Secrets: set `OPENAI_API_KEY` in `.env`; never commit `.env` or keys.
- Local LLM: ensure `ollama serve` and pull models (e.g., `ollama pull deepseek-r1:8b`).
- Cost controls: adjust `llm.openai.cost_limits` and model strategy in `config/settings.yaml`.
- Data hygiene: avoid committing `output/` and sample spreadsheets (`.gitignore` already excludes these); do not include PHI in issue logs.

