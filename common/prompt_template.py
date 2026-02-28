"""
All-to-Pipe prompt template parser.

System for parsing text templates with variable substitution from prompt objects.
"""

import re
from typing import Optional, List, Dict, Any
from ..alltopipe_types import PositivePrompt, NegativePrompt

class TemplateParser:
    """
    Parser for prompt templates with variable substitution.
    
    Supports templates with <variable> placeholders that are substituted
    from PositivePrompt and NegativePrompt attributes.
    
    Uses angle brackets <> to avoid conflicts with comfyui-dynamic-prompts module
    which uses {}, __term__, $, (), | and other symbols.
    """

    # Regex pattern for finding placeholders with <variable> syntax
    PLACEHOLDER_PATTERN = re.compile(r"<([^>]+)>")

    @staticmethod
    def find_placeholders(template: str) -> List[str]:
        """
        Find all placeholders in a template string.

        Args:
            template: Template string with <variable> placeholders

        Returns:
            List of variable names found in the template

        Raises:
            ValueError: If template is None or not a string
            
        Example:
            >>> TemplateParser.find_placeholders("A <age> person")
            ['age']
        """
        if not isinstance(template, str):
            raise ValueError("Template must be a string")

        matches = TemplateParser.PLACEHOLDER_PATTERN.findall(template)
        return matches

    @staticmethod
    def validate_template(
        template: str,
        positive_prompt: PositivePrompt,
        negative_prompt: Optional[NegativePrompt] = None,
        strict: bool = False,
    ) -> tuple[bool, List[str]]:
        """
        Validate a template against available prompt attributes.

        Args:
            template: Template string to validate (uses <variable> syntax)
            positive_prompt: PositivePrompt instance with available attributes
            negative_prompt: Optional NegativePrompt instance
            strict: If True, raise error if variable not found. If False, return as list.

        Returns:
            Tuple of (is_valid, missing_variables)
            - is_valid: True if all variables are found in prompts
            - missing_variables: List of variables not found in prompts

        Raises:
            ValueError: If strict=True and variables are missing
            
        Example:
            >>> is_valid, missing = TemplateParser.validate_template("<age> person", pos_prompt)
            >>> is_valid
            True
        """
        if not isinstance(template, str):
            raise ValueError("Template must be a string")

        placeholders = TemplateParser.find_placeholders(template)
        missing: List[str] = []

        for placeholder in placeholders:
            # Check if exists in positive prompt
            if hasattr(positive_prompt, placeholder):
                continue
            # Check if exists in negative prompt
            if negative_prompt and hasattr(negative_prompt, placeholder):
                continue
            # Variable not found
            missing.append(placeholder)

        if strict and missing:
            raise ValueError(f"Missing variables in template: {missing}")

        return (len(missing) == 0, missing)

    @staticmethod
    def parse_template(
        template: str,
        positive_prompt: PositivePrompt,
        negative_prompt: Optional[NegativePrompt] = None,
        allow_missing: bool = False,
        default_value: str = "[MISSING]",
    ) -> str:
        """
        Parse a template string with variable substitution from prompt objects.
        
        Replaces <variable> placeholders with values from prompt attributes.

        Args:
            template: Template string with <variable> placeholders
            positive_prompt: PositivePrompt instance with substitution values
            negative_prompt: Optional NegativePrompt instance
            allow_missing: If True, replace missing with default_value. If False, raise error.
            default_value: String to use for missing variables (default: "[MISSING]")

        Returns:
            Parsed template string with all variables substituted

        Raises:
            ValueError: If allow_missing=False and a variable is not found
            
        Example:
            >>> pos = PositivePrompt()
            >>> pos.age = "young"
            >>> TemplateParser.parse_template("A <age> person", pos)
            'A young person'
        Parses a template string with variable substitution from prompt objects.
        
        Replaces <variable> placeholders with values from prompt attributes.

        Args:
            template: Template string with <variable> placeholders
            positive_prompt: PositivePrompt instance with substitution values
            negative_prompt: Optional NegativePrompt instance
            allow_missing: If True, replace missing with default_value. If False, raise error.
            default_value: String to use for missing variables (default: "[MISSING]")

        Returns:
            Parsed template string with all variables substituted

        Raises:
            ValueError: If allow_missing=False and a variable is not found
            
        Example:
            >>> pos = PositivePrompt()
            >>> pos.age = "young"
            >>> TemplateParser.parse_template("A <age> person", pos)
            'A young person'
        """
        if not isinstance(template, str):
            raise ValueError("Template must be a string")

        if not template:
            return ""

        result: str = template
        placeholders = TemplateParser.find_placeholders(template)

        for placeholder in placeholders:
            # Try to get from positive prompt first
            value: Optional[str] = None
            
            if hasattr(positive_prompt, placeholder):
                value = getattr(positive_prompt, placeholder)
            elif negative_prompt and hasattr(negative_prompt, placeholder):
                value = getattr(negative_prompt, placeholder)

            if value is None:
                if allow_missing:
                    # Use default value for missing variables
                    value = default_value
                else:
                    raise ValueError(f"Variable '{placeholder}' not found in prompts")

            # Replace placeholder with value (using <> syntax)
            result = result.replace(f"<{placeholder}>", str(value))

        return result

    @staticmethod
    def parse_multiple_templates(
        templates: List[str],
        positive_prompt: PositivePrompt,
        negative_prompt: Optional[NegativePrompt] = None,
        allow_missing: bool = False,
        default_value: str = "[MISSING]",
    ) -> List[str]:
        """
        Parse multiple templates using the same prompt objects.

        Args:
            templates: List of template strings
            positive_prompt: PositivePrompt instance
            negative_prompt: Optional NegativePrompt instance
            allow_missing: If True, allow missing variables
            default_value: Default value for missing variables

        Returns:
            List of parsed strings

        Raises:
            ValueError: If any template has missing variables (when allow_missing=False)
        """
        if not isinstance(templates, list):
            raise ValueError("Templates must be a list")

        return [
            TemplateParser.parse_template(
                template,
                positive_prompt,
                negative_prompt,
                allow_missing,
                default_value,
            )
            for template in templates
        ]

    @staticmethod
    def get_required_variables(template: str) -> List[str]:
        """
        Get list of required variables from a template.

        Args:
            template: Template string with placeholders

        Returns:
            List of required variable names (no duplicates)

        Raises:
            ValueError: If template is not a string
        """
        if not isinstance(template, str):
            raise ValueError("Template must be a string")

        placeholders = TemplateParser.find_placeholders(template)
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for p in placeholders:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique
