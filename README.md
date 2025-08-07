# Medical Literature Analysis Pipeline

## Overview

This tool automatically analyzes medical literature from DOI records to extract standardized classification data for plastic surgery research. It processes large datasets (2,001+ records) from Excel spreadsheets and uses artificial intelligence to categorize research papers into three key classification areas.

- Command help: `python cli.py --help` or `python cli.py extract --help`

## Key Features

### Extracted Classification Fields

The tool extracts exactly three standardized fields from each paper:

1. **Subspecialty Focus** (Primary field) - Single category:
   - Craniofacial
   - Hand/upper limb  
   - Breast
   - Aesthetic
   - Burn
   - Generalized cutaneous disorders
   - Facial/head and neck reconstruction
   - Trunk, genital/pelvic, lower limb reconstruction

2. **Suggested Edits** (Expanded categories) - Multiple selections allowed:
   - Aesthetic / Cosmetic (non-breast)
   - Breast
   - Craniofacial
   - Hand/Upper extremity & Peripheral Nerve
   - Burn
   - Generalized Cutaneous Disorders
   - Head & Neck Reconstruction
   - Trunk / Pelvic / Lower-Limb Reconstruction
   - Gender-affirming Surgery
   - Education & Technology
   - Research Methods & Statistics

3. **Priority Topics** (Community priorities) - Multiple selections with detailed sub-items:
   - Patient Safety & Clinical Standards
   - Technology and Innovation
   - Diversity, Equity, and Inclusion
   - Education and Training
   - Global Health and Access
   - Research and Evidence
   - Professional Development
   - Regulatory and Policy

### Auditability Features

- **Confidence scoring** for each classification
- **Complete audit trails** for research reproducibility
- **Human review flags** for uncertain cases
- **Batch processing** of thousands of records
- **Quality validation** and error reporting
- **Multiple export formats** (YAML for research, CSV/Excel for analysis)

## Prerequisites

### Required Software

You'll need to install these programs on your computer:

1. **Python 3.9 or newer**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Git** (for downloading the tool)
   - Download from: https://git-scm.com/downloads

3. **Ollama** (for local AI processing) - RECOMMENDED
   - Download from: https://ollama.ai/
   - After installation, run: `ollama pull deepseek-r1:8b`

4. **Alternative: OpenAI API** (cloud-based AI)
   - Sign up at: https://platform.openai.com/
   - Get API key and set as environment variable `OPENAI_API_KEY`
   - Note: Costs apply for API usage but results may be more accurate

### Data Requirements

- **Excel file (.xlsx)** with your literature records
- **Required column: "DOI"** - Each row must have a DOI number  
- **Required column: "Title"** - Paper titles are essential for classification
- **Highly recommended: "Abstract Note"** - Abstracts significantly improve accuracy (60% vs 35% max confidence)
- **Recommended columns:** Author, Publication Year, Journal, Author Affiliations

## Quick Start Guide

### Step 1: Download and Setup

```bash
# Download the tool
git clone https://github.com/your-repo/med-doi-scraping.git
cd med-doi-scraping

# Create isolated environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

1. Place your Excel file in the project folder
2. Ensure it has columns named "DOI" and "Abstract Note"
3. The tool works with any Excel format but expects these specific column names

### Step 3: Test with One Record

```bash
# Test extraction on a single record
python cli.py test
# See data quality summary
python cli.py preview
```

This shows:

- Total records in your file
- How many have DOIs and abstracts
- Sample records to verify format

### Step 4: Run Extraction

```bash
# Interactive mode - you'll be prompted for all options
python cli.py extract

# Or specify parameters directly
python cli.py extract --skip 0 --limit 10 --provider ollama
```

## How to Use

### Data Privacy and Security

- **Local Processing:** When using Ollama, all data stays on your computer
- **Cloud Processing:** OpenAI receives abstracts and doi in prompt.


### Interactive Mode (Recommended for First Time)

Simply run:
```bash
python cli.py extract
```

The tool will ask you:
1. Which Excel file to process
2. How many records to skip (0 for start from beginning)
3. How many records to process (blank for all)
4. Which AI provider to use (ollama recommended)
5. Whether to reprocess existing results
6. How many records to process simultaneously

### Common Workflows

```bash
# Process records 101-200 using local AI
python cli.py extract --skip 100 --limit 100 --provider ollama

# Process first 50 records using OpenAI
python cli.py extract --skip 0 --limit 50 --provider openai --batch-size 3

# Process all remaining records, force reprocess existing
python cli.py extract --force

# Small Test Run (10 records)
python cli.py extract --skip 0 --limit 10

# Large Batch (500 records at a time)
python cli.py extract --skip 0 --limit 500
python cli.py extract --skip 500 --limit 500
python cli.py extract --skip 1000 --limit 500

# Resume After Interruption (Check what failed and retry)
python cli.py retry

# Batch Size Optimization for concurrency - Check system requirements if local as you will need ALOT of RAM
# Conservative (slower but more reliable)
python cli.py extract --batch-size 1

# Aggressive (faster but may hit rate limits)  
python cli.py extract --batch-size 10
```

## Understanding Results

### Output Structure

The tool creates several folders:
```
output/
├── extracted/          # Individual classification results (YAML files)
├── logs/              # Processing logs and detailed interactions  
└── failures/          # Records that couldn't be processed
```

### Individual Results (YAML files)

Each successfully processed paper gets a file like `10.1097_PRS.0000000000004601.yaml`:

```yaml
doi: 10.1097/PRS.0000000000004601
title: "A Prospective Evaluation of Three-Dimensional Image Simulation..."
subspecialty_focus: Breast
suggested_edits:
  - Education & Technology  
  - Breast
priority_topics:
  - Patient Safety & Clinical Standards
  - Technology and Innovation
confidence_scores:
  overall: 0.82
human_review_required: false
```

### Confidence Scoring System

The tool uses a sophisticated confidence scoring system based on the quality and completeness of available input data. Confidence scores help medical researchers understand how reliable each classification is.

#### Input Field Impact on Maximum Confidence

The system weights confidence based on field availability, recognizing that medical classification accuracy depends heavily on having sufficient context:

| Available Fields | Maximum Confidence | Reasoning |
|-----------------|-------------------|-----------|
| **Title only** | 35% | Minimal context for accurate medical classification |
| **Title + Abstract** | 60% | Abstract provides core medical content for classification |
| **+ Authors/Affiliations** | 75% | Institutional context helps identify subspecialty focus |
| **+ Publication details** | 85% | Journal, year, DOI add credibility and context |
| **+ Complete metadata** | 100% | Full dataset enables highest confidence classifications |

#### Confidence Score Interpretation

- **0.8-1.0:** High confidence, likely accurate - Use without review
- **0.6-0.8:** Medium confidence, review recommended - Check for obvious errors
- **Below 0.6:** Low confidence, human review required - Manual verification needed

#### Special Cases

- **Missing Abstract:** Classifications proceed but confidence is capped at 60% maximum
- **"Other" Selections:** Confidence reduced by 20-30% unless specific explanations provided  
- **Multiple Warnings:** Each missing field reduces confidence by ~5%
- **Title Only Records:** Processed with 35% maximum confidence for basic categorization

This transparency allows researchers to make informed decisions about which classifications to trust and which require manual review.

### Session Logs

Each run creates a session log showing:

- Total records processed
- Success/failure counts
- Processing time
- Error categories

## Data Export

### Export to Excel/CSV

```bash
# Export all results to CSV
python cli.py export

# Export to Excel format
python cli.py export --format excel

# Custom filename and location
python cli.py export --format excel --output-file my_results.xlsx
```

The exported file includes:

- All classification fields
- Confidence scores
- Processing metadata
- Quality flags

### Quality Validation

```bash
# Generate quality report
python cli.py validate
```

Shows:

- Confidence score distribution
- Field coverage statistics
- Human review requirements
- Data quality metrics

## Troubleshooting

### Common Issues

**"No abstracts found"** (Warning, not error)

- Impact: Processing continues but confidence is limited to 60% maximum  
- Solution: For higher accuracy, provide abstracts in "Abstract Note" column
- Note: Records without abstracts are still processed for basic categorization

**"Connection refused" with Ollama**

- Solution: Make sure Ollama is running: `ollama serve`
- Or switch to OpenAI: `--provider openai`

**"Rate limit exceeded" with OpenAI**

- Solution: Reduce batch size: `--batch-size 1`
- Or use Ollama for unlimited local processing

**Processing is very slow**

- With Ollama: Normal, especially first time. Each record takes 30-60 seconds
- With OpenAI: Should be faster, check your internet connection

**Python/pip not found**

- Solution: Reinstall Python and check "Add to PATH" during installation
- Or use full path: `/usr/bin/python3` instead of `python`

### Getting Help

1. **Check the logs:** Files in `output/logs/` show detailed error messages
2. **Review failures:** Files in `output/failures/` list what couldn't be processed
3. **Test with single record:** `python cli.py test` to isolate issues
4. **Retry failed extractions:** `python cli.py retry` to reprocess failures

### Error Categories

- **missing_data:** Critical fields missing (title/DOI) - cannot proceed
- **llm_error:** AI service temporarily unavailable (retry recommended)  
- **parsing_error:** AI response couldn't be understood (retry recommended)
- **timeout:** Processing took too long (retry with different provider)

**Note:** Missing abstracts now generate warnings (reduced confidence) rather than errors.


---

**Version:** 1.0  
**Last Updated:** August 2025  
**Tested With:** Python 3.9+, Ollama deepseek-r1:8b, OpenAI GPT-4o-mini