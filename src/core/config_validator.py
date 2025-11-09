"""Configuration schema validation for medical DOI feature extraction."""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration structure and required fields."""

    # Required top-level keys
    REQUIRED_KEYS = ['llm', 'processing']

    # Nested structure requirements
    NESTED_REQUIREMENTS = {
        'processing': ['batch_size', 'strict_validation'],
        'llm': []  # At least one provider should be configured
    }

    # Optional but commonly used keys
    OPTIONAL_KEYS = {
        'processing': ['test_mode', 'test_limit', 'delay_between_requests', 'force_reprocess'],
        'audit': ['enabled', 'output_dir'],
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        errors = []

        # Check required top-level keys
        for key in cls.REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required top-level key: '{key}'")

        # Check nested requirements
        for parent_key, required_children in cls.NESTED_REQUIREMENTS.items():
            if parent_key in config:
                if not isinstance(config[parent_key], dict):
                    errors.append(f"Key '{parent_key}' must be a dictionary")
                    continue

                for child_key in required_children:
                    if child_key not in config[parent_key]:
                        errors.append(
                            f"Missing required nested key: '{parent_key}.{child_key}'"
                        )
            else:
                # Parent key missing, already reported above
                pass

        # Validate LLM provider configuration
        if 'llm' in config and isinstance(config['llm'], dict):
            llm_config = config['llm']
            available_providers = []

            if 'ollama' in llm_config:
                available_providers.append('ollama')
            if 'openai' in llm_config:
                available_providers.append('openai')

            if not available_providers:
                errors.append(
                    "No LLM providers configured. Add 'ollama' or 'openai' under 'llm' section"
                )
            else:
                logger.info(f"Configured LLM providers: {', '.join(available_providers)}")

        # Validate processing config types
        if 'processing' in config and isinstance(config['processing'], dict):
            processing = config['processing']

            # Check batch_size is positive integer
            if 'batch_size' in processing:
                if not isinstance(processing['batch_size'], int) or processing['batch_size'] <= 0:
                    errors.append("'processing.batch_size' must be a positive integer")

            # Check strict_validation is boolean
            if 'strict_validation' in processing:
                if not isinstance(processing['strict_validation'], bool):
                    errors.append("'processing.strict_validation' must be a boolean")

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def validate_and_raise(cls, config: Dict[str, Any]) -> None:
        """
        Validate configuration and raise ValueError if invalid.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        is_valid, errors = cls.validate(config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)
        logger.info("Configuration validation passed")

    @classmethod
    def get_nested_value(cls, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Safely get nested configuration value.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., 'processing.strict_validation')
            default: Default value if path doesn't exist

        Returns:
            Configuration value or default

        Example:
            >>> get_nested_value(config, 'processing.strict_validation', False)
            True
        """
        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value
