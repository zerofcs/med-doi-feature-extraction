"""
Function tools for the extraction pipeline generator agent.

These tools allow the agent to interact with the filesystem, validate configurations,
and generate preview content.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
from agents import function_tool


@function_tool
def validate_field_definition(
    field_name: str,
    field_type: str,
    allowed_values: Optional[List[str]] = None
) -> Dict:
    """
    Validate a field definition structure.

    Args:
        field_name: Name of the field (e.g., "funding_agency")
        field_type: Type of field ("text", "enum", "numeric", "boolean")
        allowed_values: List of allowed values for enum types

    Returns:
        dict with validation results and suggestions
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }

    # Validate field name
    if not field_name or not field_name.replace('_', '').isalnum():
        results["valid"] = False
        results["errors"].append("Field name must be alphanumeric with underscores")

    if field_name and not field_name.islower():
        results["warnings"].append("Field name should be lowercase (snake_case recommended)")

    # Validate field type
    valid_types = ["text", "enum", "numeric", "boolean"]
    if field_type not in valid_types:
        results["valid"] = False
        results["errors"].append(f"Field type must be one of: {', '.join(valid_types)}")

    # Validate enum values
    if field_type == "enum":
        if not allowed_values or len(allowed_values) == 0:
            results["valid"] = False
            results["errors"].append("Enum fields must have at least one allowed value")
        elif len(allowed_values) > 100:
            results["warnings"].append("Large number of allowed values - consider using 'text' type")

        # Check if "Other" is included
        if allowed_values and "Other" not in allowed_values:
            results["suggestions"].append("Consider adding 'Other' option for unexpected values")

    return results


@function_tool
def preview_prompt_with_example(
    prompt_template: str,
    example_data: Dict
) -> str:
    """
    Show how a prompt template will look when filled with example data.

    Args:
        prompt_template: Prompt template with {placeholders}
        example_data: Dictionary of example values to fill in

    Returns:
        Rendered prompt with example data
    """
    try:
        rendered = prompt_template.format(**example_data)
        return f"PREVIEW:\n{'=' * 60}\n{rendered}\n{'=' * 60}"
    except KeyError as e:
        return f"ERROR: Missing placeholder in example data: {e}"
    except Exception as e:
        return f"ERROR: {str(e)}"


@function_tool
def check_existing_extractor(name: str) -> Dict:
    """
    Check if an extractor with the given name already exists.

    Args:
        name: Name of the extractor to check (e.g., "doi", "country", "funding")

    Returns:
        dict with existence status and details
    """
    base_path = Path(".")

    results = {
        "exists": False,
        "name": name,
        "files_found": [],
        "can_fork": False,
        "can_modify": False
    }

    # Check for field config
    field_path = base_path / "config" / "fields" / f"{name}_fields.yaml"
    if field_path.exists():
        results["files_found"].append(str(field_path))
        results["exists"] = True

    # Check for prompt config
    prompt_path = base_path / "config" / "prompts" / f"{name}_prompts.yaml"
    if prompt_path.exists():
        results["files_found"].append(str(prompt_path))
        results["exists"] = True

    # Check for extractor implementation
    extractor_path = base_path / "src" / "extractors" / f"{name}_extractor.py"
    if extractor_path.exists():
        results["files_found"].append(str(extractor_path))
        results["exists"] = True

    # Check for legacy extractor.py (for DOI)
    if name == "doi" and (base_path / "src" / "extractor.py").exists():
        results["files_found"].append("src/extractor.py")
        results["exists"] = True

    results["can_fork"] = results["exists"]
    results["can_modify"] = results["exists"]

    return results


@function_tool
def list_available_base_classes() -> List[Dict]:
    """
    List extractor base classes that can be used.

    Returns:
        List of available base classes with descriptions
    """
    base_path = Path("src/extractors")

    base_classes = [
        {
            "name": "BaseExtractor",
            "file": "src/extractors/base.py",
            "description": "Abstract base class with template methods for extraction pipeline",
            "exists": (base_path / "base.py").exists()
        },
        {
            "name": "DOIExtractor",
            "file": "src/extractors/doi_extractor.py",
            "description": "Complete example for medical literature classification",
            "exists": (base_path / "doi_extractor.py").exists()
        },
        {
            "name": "CountryExtractor",
            "file": "src/extractors/country_extractor.py",
            "description": "Example for author affiliation extraction",
            "exists": (Path("src/extractors/country_extractor.py")).exists()
        }
    ]

    return [bc for bc in base_classes if bc["exists"]]


@function_tool
def generate_field_yaml(fields: Dict[str, List[str]]) -> str:
    """
    Generate YAML content for field definitions.

    Args:
        fields: Dictionary mapping field names to lists of allowed values
               For non-enum fields, use empty list

    Returns:
        YAML formatted string
    """
    try:
        yaml_content = yaml.dump(fields, default_flow_style=False, sort_keys=False)
        return yaml_content
    except Exception as e:
        return f"ERROR generating YAML: {str(e)}"


@function_tool
def generate_prompt_yaml(prompts: Dict[str, str]) -> str:
    """
    Generate YAML content for prompt templates.

    Args:
        prompts: Dictionary with 'system' and 'extraction' prompt templates

    Returns:
        YAML formatted string
    """
    try:
        if "system" not in prompts:
            return "ERROR: Missing 'system' prompt"
        if "extraction" not in prompts:
            return "ERROR: Missing 'extraction' prompt"

        yaml_content = yaml.dump(prompts, default_flow_style=False, sort_keys=False,
                                 default_style='|', width=80)
        return yaml_content
    except Exception as e:
        return f"ERROR generating YAML: {str(e)}"


@function_tool
def save_config_file(file_path: str, content: str, require_approval: bool = True) -> Dict:
    """
    Save a configuration file to disk.

    Args:
        file_path: Relative path to save the file
        content: File content to write
        require_approval: Whether this requires user approval (always True for safety)

    Returns:
        dict with save status
    """
    # NOTE: This is a placeholder - actual file writing happens in the CLI
    # with explicit user approval. The agent just returns the intent.

    return {
        "action": "save_file",
        "path": file_path,
        "content_preview": content[:200] + "..." if len(content) > 200 else content,
        "full_content": content,
        "requires_approval": True,
        "status": "pending_approval"
    }


@function_tool
def list_existing_extractors() -> List[str]:
    """
    List all existing extraction pipelines.

    Returns:
        List of extractor names that exist in the system
    """
    extractors = []

    fields_dir = Path("config/fields")
    if fields_dir.exists():
        for field_file in fields_dir.glob("*_fields.yaml"):
            name = field_file.stem.replace("_fields", "")
            extractors.append(name)

    return sorted(set(extractors))


@function_tool
def load_existing_config(extractor_name: str, config_type: str) -> Dict:
    """
    Load existing configuration for an extractor.

    Args:
        extractor_name: Name of the extractor (e.g., "doi", "country")
        config_type: Type of config ("fields" or "prompts")

    Returns:
        dict with configuration content or error
    """
    if config_type not in ["fields", "prompts"]:
        return {"error": "config_type must be 'fields' or 'prompts'"}

    file_path = Path(f"config/{config_type}/{extractor_name}_{config_type}.yaml")

    if not file_path.exists():
        return {"error": f"Configuration file not found: {file_path}"}

    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)

        return {
            "name": extractor_name,
            "type": config_type,
            "path": str(file_path),
            "content": content
        }
    except Exception as e:
        return {"error": f"Failed to load config: {str(e)}"}
