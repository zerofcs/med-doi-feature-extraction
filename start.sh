#!/usr/bin/env bash
source venv/bin/activate
python cli.py run -c config/pipelines/screen-fat-grafting.yaml -i data/data-abstracts.csv
