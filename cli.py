#!/usr/bin/env python3
"""
Convenience entry point to run the config-driven CLI.

Usage examples:
  python cli.py validate --config pipelines/doi.yaml
  python cli.py run --config pipelines/doi.yaml --input data/final_test.csv --id-column DOI --limit 5
"""

from src.cli import main


if __name__ == "__main__":
    main()

