Note: This document describes the full feature set as designed. The current CLI build includes `test`, `run`, `validate`, and `retry`. Commands like `preview`, `benchmark`, `export`, and the AI generator referenced below may not be present in this build.
### Feature Breakdown: `cli.py`

This script serves as the main entry point and user interface for the entire medical literature extraction pipeline. It is built using the `Typer` library for creating robust command-line applications and `rich` for providing a polished, user-friendly console experience.

---

### 1. Core Application & Framework

#### 1.1. Typer CLI Application
-   **Implementation:** `app = typer.Typer(...)`
-   **Feature:** The entire application is structured as a `Typer` app. This provides automatic help generation (`-h`, `--help`), command discovery, and type-hinted command-line argument parsing.

#### 1.2. Rich Console UI
-   **Implementation:** `console = Console()` and widespread use of `console.print`, `Table`, `Progress`, `Panel`, and `Prompt`.
-   **Feature:** The application avoids plain text output in favor of a rich user interface with colors, tables, progress bars, and formatted panels. This makes the tool more intuitive and professional.

#### 1.3. Environment Variable Loading
-   **Implementation:** `from dotenv import load_dotenv; load_dotenv()`
-   **Feature:** The application automatically loads environment variables from a `.env` file at startup. This is primarily used for securely managing secrets like `OPENAI_API_KEY` without hardcoding them.

---

### 2. Configuration Management

#### 2.1. Layered YAML Configuration
-   **Implementation:** `load_config()` function.
-   **Feature:** The system uses a sophisticated layered configuration model. It loads a `settings.base.yaml` file for global defaults and then merges a pipeline-specific overlay (e.g., `settings.doi.yaml`) on top. This allows for shared settings across pipelines while permitting specific overrides.

#### 2.2. Dynamic Config Selection
-   **Implementation:** `_select_config()` and the use of the `MED_CONFIG_PATH` environment variable.
-   **Feature:** The application is not hardcoded to a single configuration file. Users can choose a config interactively from a menu, or specify one directly via an environment variable. This makes it easy to switch between different extraction pipelines (e.g., DOI vs. Country).

#### 2.3. Recursive Dictionary Merging
-   **Implementation:** `_deep_update(base: dict, override: dict)`
-   **Feature:** This helper function ensures that when the overlay config is merged with the base, nested dictionaries are updated recursively instead of being replaced entirely. This is crucial for allowing fine-grained overrides (e.g., changing just one model's temperature without redefining the entire `llm` block).

---

### 3. Interactive Shell & User Experience (UX)

#### 3.1. Main Interactive Menu
-   **Implementation:** `@app.callback(invoke_without_command=True)` and the `entry(ctx)` function.
-   **Feature:** If the script is run without any command, it presents an interactive main menu. This serves as a guided entry point for users who may not know the specific commands, improving usability.

#### 3.2. Context-Aware Action Menu
-   **Implementation:** `_interactive_actions()` function.
-   **Feature:** After a configuration is selected, the application presents a secondary menu of relevant actions. It intelligently detects the current pipeline (DOI vs. Country) and only shows commands that apply to it, preventing user confusion.

#### 3.3. Reusable Interactive Prompts
-   **Implementation:** `_choose_from_list()`, `Confirm.ask`, `IntPrompt.ask`.
-   **Feature:** The script uses standardized helper functions for common UI patterns like selecting from a numbered list or confirming an action. This ensures a consistent interactive experience across all commands.

---

### 4. Data Handling & Preparation

#### 4.1. Excel Data Loading
-   **Implementation:** `load_excel_data(...)` function.
-   **Feature:** This function is the primary data ingestion point for DOI-related pipelines.
    -   It uses the `pandas` library for robust Excel file reading.
    -   It deserializes each row into a structured `InputRecord` Pydantic model, providing data validation at the source.
    -   It gracefully handles missing values (`NaN`).
    -   It filters the dataset to only include records that have a DOI, which is a core business rule.

#### 4.2. Windowed Processing (Skip/Limit)
-   **Implementation:** The `skip` and `limit` parameters in `load_excel_data` and various commands.
-   **Feature:** Users can process large files in manageable chunks by specifying a starting offset (`--skip`) and a maximum number of records to process (`--limit`). This is essential for testing, debugging, and resuming large jobs.

---

### 5. Primary Commands (The Pipelines)

#### 5.1. `preview` Command
-   **Purpose:** Inspect the input Excel file before running a full extraction.
-   **Features:**
    -   Displays summary statistics (total rows/columns).
    -   Generates a **Data Quality Summary** table, showing the percentage of non-empty values for critical fields like `DOI` and `Abstract Note`.
    -   Prints a formatted preview of the first few records that contain a DOI.

#### 5.2. `test` Command
-   **Purpose:** Run a full extraction on a single record for debugging and verification.
-   **Features:**
    -   Loads only one record (respecting `--skip`).
    -   Allows interactive or command-line overrides for the LLM provider and model, making it easy to test different configurations.
    -   Instantiates the full `DOIExtractor` and `AuditLogger` to simulate a real run.
    -   Prints a detailed, formatted breakdown of the extracted fields, confidence scores, and any warnings or validation flags.

IMPORTANT: 5.3 and 5.4 are inferred via the config yaml they are NOT HARDCODED IN ANYWAY
#### 5.3. `extract-doi` Command (Core DOI Pipeline)
-   **Purpose:** The main workhorse command for running the medical literature classification pipeline.
-   **Features:**
    -   **Full Interactivity:** If run without flags, it interactively prompts the user for all necessary parameters (file path, skip/limit, provider, model, etc.).
    -   **Batch Processing:** It processes records in concurrent batches (`--batch-size`) using `asyncio` for high performance.
    -   **Rich Progress Bars:** Displays three separate progress bars in real-time to track:
        1.  Overall progress.
        2.  Records with reduced confidence (e.g., due to a missing abstract).
        3.  Total failed extractions.
    -   **Confirmation Prompt:** Asks for user confirmation before starting a large job to prevent accidental runs.
    -   **Comprehensive Summary:** After completion, it displays a final summary table with key metrics from the `AuditLogger`'s session summary.

#### 5.4. `extract-country` Command (Country Extraction Pipeline)
-   **Purpose:** Runs a separate, distinct pipeline to extract author country from affiliation text.
-   **Features:**
    -   **Demonstrates Multi-Pipeline Architecture:** It imports and uses a different engine (`CountryExtractionEngine`), showing the system's extensibility.
    -   **Progressive CSV Output:** It saves results incrementally to a single CSV file after each batch. This is a crucial feature for resilience, ensuring that progress is not lost if the run is interrupted.

#### 5.5. `benchmark` Command
-   **Purpose:** A powerful utility to compare the performance, cost, and quality of different LLM models.
-   **Features:**
    -   Runs the same sample of data against a list of specified models.
    -   Collects detailed metrics: success rate, average confidence, processing time (min/max/median), and total cost.
    -   Presents a final comparison table for easy analysis.
    -   Provides high-level recommendations based on the results (e.g., which model is best for cost, balance, or accuracy).
    -   Saves the full, detailed benchmark results to a timestamped YAML file for later review.

#### 5.6. `retry` Command
-   **Purpose:** Intelligently retry failed extractions from previous runs.
-   **Features:**
    -   Loads failed records from the `output/failures/` directory.
    -   Allows filtering by a specific `session-id`.
    -   Groups failures by category and applies predefined retry strategies (e.g., using a fallback provider for an `llm_error`).
    -   Tracks retry attempts to avoid infinite loops.

#### 5.7. `export` Command
-   **Purpose:** Aggregate the individual YAML output files into a single spreadsheet.
-   **Features:**
    -   Scans a directory for all extracted YAML files.
    -   **Data Flattening:** It intelligently flattens the nested YAML structure, creating separate columns for each extracted field, confidence score, and key piece of transparency metadata (like model used, cost, and processing time).
    -   Supports both CSV and Excel as output formats.
    -   Provides a summary of the exported data's quality (e.g., number of high-confidence records).

#### 5.8. `validate` Command
-   **Purpose:** Perform a post-run quality analysis on a directory of extracted data.
-   **Features:**
    -   Scans the output directory and calculates aggregate quality metrics.
    -   Generates a **Quality Validation Report** showing the distribution of confidence scores.
    -   Calculates and displays **Field Coverage** statistics, showing how many records had each target field successfully extracted.

---

### 6. Pipeline Generation & Extensibility

#### 6.1. AI-Powered Generator (`generate-extractor`)
-   **Purpose:** An advanced tool to bootstrap the configuration for a new extraction pipeline.
-   **Features:**
    -   **Conversational AI:** It uses an AI agent (via the `agents` library) to have a conversation with the developer about their extraction needs.
    -   **Multi-Mode Operation:** Supports creating a new pipeline from scratch, modifying an existing one, or forking an existing one as a template.
    -   **Code Generation:** The agent generates the necessary YAML configuration files (for fields and prompts) and provides Python code snippets for the new `Extractor` class and CLI command.

#### 6.2. Fallback Config Wizard
-   **Implementation:** `_run_config_wizard()`
-   **Feature:** This is a crucial fallback for the AI generator. If the `OPENAI_API_KEY` is not set or the agent fails, it launches a local, interactive, step-by-step wizard that asks the user for the same information (fields, prompts) and generates the configuration files. This ensures the generation feature is always available.
