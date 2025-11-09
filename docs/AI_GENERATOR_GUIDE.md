# AI-Powered Extraction Pipeline Generator

## Overview

The AI-Powered Extraction Pipeline Generator is an interactive tool that helps researchers create new extraction pipelines without writing code. Using OpenAI's Agents SDK, it provides a conversational interface to design and generate extraction configurations.

## Features

- **Conversational Interface**: Natural language conversation to understand requirements
- **Guided Workflow**: Step-by-step guidance through the extraction design process
- **Intelligent Validation**: Automatic validation of field definitions and prompts
- **Configuration Generation**: Automated creation of YAML configuration files
- **Code Templates**: Provides ready-to-use code templates for implementation
- **Multiple Modes**: Create new, modify existing, or fork extractors

## Prerequisites

1. **OpenAI API Key**: Required for the AI agent
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Dependencies**: Already installed with the project
   - `openai-agents` - OpenAI Agents SDK
   - `pydantic` - Data validation
   - `pyyaml` - YAML processing

## Usage

### Create New Extractor

Start a conversation to create a brand new extraction pipeline:

```bash
python -m src.cli generate-extractor
```

Or specify the name upfront:

```bash
python -m src.cli generate-extractor --name funding
```

### Modify Existing Extractor

Update field definitions or prompts for an existing extractor:

```bash
python -m src.cli generate-extractor --modify doi
python -m src.cli generate-extractor --modify country
```

### Fork Existing Extractor

Create a new extractor based on an existing one:

```bash
python -m src.cli generate-extractor --fork country --name institution
python -m src.cli generate-extractor --fork doi --name clinical-trials
```

## Workflow

### Phase 1: Discovery

The AI agent asks clarifying questions to understand your requirements:

```
AI: What kind of data are you working with?
You: Medical papers with funding information

AI: What's your data source format?
You: Excel file with DOI, Title, Abstract, and Acknowledgments columns

AI: What specific fields do you want to extract?
You: Funding agency, grant number, funding amount, and funding type
```

### Phase 2: Validation

The agent validates your requirements using built-in tools:

- Checks field naming conventions
- Validates field types (text, enum, numeric, boolean)
- Verifies allowed values for enum fields
- Suggests improvements (e.g., adding "Other" option)

### Phase 3: Plan Presentation

The agent presents a comprehensive plan:

```
NEW EXTRACTOR: "funding-extractor"

FIELDS:
✓ funding_agency (text)
✓ grant_number (text)
✓ funding_amount (numeric)
✓ funding_type (enum: Government/Private/Mixed/Other)

PROMPTS:
✓ System: Expert at extracting funding information
✓ Extraction: Parse acknowledgments for funding details

FILES TO CREATE:
- config/fields/funding_fields.yaml
- config/prompts/funding_prompts.yaml

Does this plan look good? (yes/modify/cancel)
```

### Phase 4: Configuration Generation

Upon approval, the agent generates configuration files:

1. **Field Definitions** (`config/fields/{name}_fields.yaml`):
   ```yaml
   funding_agency:
     - "National Institutes of Health"
     - "National Science Foundation"
     - "Other"

   funding_type:
     - "Government"
     - "Private"
     - "Mixed"
     - "Other"
   ```

2. **Prompt Templates** (`config/prompts/{name}_prompts.yaml`):
   ```yaml
   system: |
     You are an expert at extracting funding information from academic acknowledgments.
     Your task is to identify funding agencies, grant numbers, and amounts.

   extraction: |
     Analyze the following acknowledgments and extract funding information:

     Text: {acknowledgments}

     Extract: {funding_agency}, {grant_number}, {funding_amount}, {funding_type}
   ```

### Phase 5: Implementation Guidance

The agent provides next steps with code templates:

1. **Extractor Class Template** - Ready-to-customize Python class
2. **CLI Command Snippet** - Code to add to `src/cli.py`
3. **Testing Instructions** - How to test the new pipeline

## Agent Tools

The AI agent has access to these function tools:

### Validation Tools
- `validate_field_definition()` - Validate field structure and naming
- `preview_prompt_with_example()` - Show rendered prompts with sample data

### Discovery Tools
- `check_existing_extractor()` - Check if extractor exists
- `list_available_base_classes()` - List usable base classes
- `list_existing_extractors()` - List all extractors in system
- `load_existing_config()` - Load existing configuration

### Generation Tools
- `generate_field_yaml()` - Generate field YAML content
- `generate_prompt_yaml()` - Generate prompt YAML content
- `save_config_file()` - Save configuration file (with user approval)

## Example Session

### Creating a Funding Extractor

```bash
$ python -m src.cli generate-extractor --name funding

AI Extraction Pipeline Generator
Powered by OpenAI Agents

Mode: new
Initial message: I want to create a new extraction pipeline called 'funding'

AI: Hi! I'll help you create the 'funding' extraction pipeline.
    Let me start by understanding your requirements.

    What type of data will this extractor process?

You: Medical research papers - I need to extract funding information

AI: Great! What is your data source? For example:
    - Excel file with specific columns
    - CSV file
    - Database records

You: Excel file with columns: DOI, Title, Abstract, Acknowledgments

AI: Perfect. What specific funding-related fields do you want to extract?

You: I need:
    - Funding agency name
    - Grant number
    - Funding amount (if mentioned)
    - Type of funding (government vs private)

AI: Excellent! Can you provide an example of an Acknowledgments section?
    This will help me design better extraction prompts.

You: "This work was supported by NIH grant R01-CA123456 ($500,000)
     and the Pfizer Foundation ($100,000)."

AI: [Using tool: validate_field_definition]
    [Using tool: preview_prompt_with_example]

    Thank you! Here's the plan I've created:

    [Shows detailed plan...]

    Does this plan look good? (yes/modify/cancel)

You: yes

AI: Great! Let me generate the configuration files...

    [Generates config/fields/funding_fields.yaml]
    [Shows preview]

    Approve this file? (yes/edit/skip)

You: yes

AI: ✓ Created config/fields/funding_fields.yaml

    [Generates config/prompts/funding_prompts.yaml]
    [Shows preview]

    Approve this file? (yes/edit/skip)

You: yes

AI: ✓ Created config/prompts/funding_prompts.yaml

    SUCCESS! Your extraction pipeline is configured.

    Next steps:
    1. Create extractor class: src/extractors/funding_extractor.py
       [Shows template code]
    2. Add CLI command in src/cli.py
       [Shows code snippet]
    3. Test: python -m src.cli extract-funding --file data.xlsx
```

## Architecture

```
src/agents/
├── __init__.py
├── tools.py                    # Function tools for agent
├── config_generator.py         # Config file generation utilities
└── extractor_generator.py      # Main agent orchestrator

Generated files:
config/
├── fields/{name}_fields.yaml    # Field definitions
└── prompts/{name}_prompts.yaml  # Prompt templates

Implementation (manual):
src/extractors/{name}_extractor.py   # Extractor class
src/cli.py                            # CLI command
```

## Benefits

### For Researchers
- **No coding required** for configuration
- **Guided process** reduces errors
- **Examples and suggestions** improve quality
- **Validation** catches issues early

### For Developers
- **Standardized configurations** across extractors
- **Reusable templates** speed up development
- **Documented patterns** improve maintainability
- **Extensible** - easy to add new tools

## Limitations

### What the AI Generates
- ✓ Field definitions (YAML)
- ✓ Prompt templates (YAML)
- ✓ Code templates (for reference)
- ✓ CLI command snippets (for reference)

### What Requires Manual Implementation
- ✗ Python extractor class (template provided)
- ✗ CLI command integration (snippet provided)
- ✗ Data model classes (template provided)
- ✗ Testing code

### Why Not Full Code Generation?
- **Safety**: Code generation can introduce bugs
- **Customization**: Every extractor has unique requirements
- **Learning**: Researchers understand the system better
- **Flexibility**: Easy to customize generated configurations

## Troubleshooting

### "OPENAI_API_KEY not set"
Set your OpenAI API key:
```bash
export OPENAI_API_KEY='sk-...'
```

### Agent doesn't understand requirements
Provide more specific examples:
- Show sample input data
- Provide expected output examples
- Clarify edge cases

### Generated prompts don't work well
- Modify the prompts.yaml file manually
- Provide more detailed examples to the agent
- Iterate on the prompts with test data

### Want to modify generated config
Simply edit the YAML files directly:
- `config/fields/{name}_fields.yaml`
- `config/prompts/{name}_prompts.yaml`

## Best Practices

1. **Provide Examples**: Real data examples help the agent create better prompts
2. **Be Specific**: Clearly define field types and allowed values
3. **Start Simple**: Create basic extractor first, then enhance
4. **Iterate**: Test and refine configurations based on results
5. **Document**: Add comments to generated YAML files for future reference

## Future Enhancements

Potential improvements for the AI generator:

- **Full code generation**: Generate complete extractor classes
- **Interactive testing**: Test extractions during configuration
- **Version control**: Track changes to configurations
- **Prompt optimization**: AI-suggested prompt improvements based on test results
- **Multi-agent system**: Separate agents for fields, prompts, and code
- **Integration with existing data**: Auto-detect fields from sample files

## See Also

- [OpenAI Agents SDK Documentation](https://github.com/anthropics/openai-agents)
- [Extractor Architecture Guide](EXTRACTOR_ARCHITECTURE.md)
- [Field Configuration Reference](FIELD_CONFIG_REFERENCE.md)
- [Prompt Engineering Best Practices](PROMPT_ENGINEERING.md)
