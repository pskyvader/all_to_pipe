"""
All-to-Pipe template parser node.

Parses template strings using current prompt data.
"""

from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe
from ..common.utils import deep_copy_pipe
from ..common.prompt_template import TemplateParser

#TODO: parsing should be internal and only at export nodes
# so this node is unnecessary 
class TemplateParserNode:
    """
    Parses template strings using available prompt data.
    
    Takes a template with {variable} placeholders and substitutes
    values from the current PositivePrompt and NegativePrompt objects.
    
    Returns the parsed string result.
    """

    def __init__(self) -> None:
        """Initialize the template parser node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe,
        template: str,
        allow_missing: bool = False,
        default_value: str = "[MISSING]",
    ) -> Tuple[str]:
        """
        Execute the node and parse a template using prompt data.

        Args:
            pipe: The input Pipe instance with prompt data
            template: Template string with {variable} placeholders
            allow_missing: If True, replace missing variables with default_value
                           If False, raise error on missing variables
            default_value: String to use for missing variables (default: "[MISSING]")

        Returns:
            Tuple containing the parsed string

        Raises:
            ValueError: If template has missing variables (when allow_missing=False)
        """
        if not template or not template.strip():
            return ("",)

        try:
            parsed: str = TemplateParser.parse_template(
                template,
                pipe.positive_prompt,
                pipe.negative_prompt,
                allow_missing=allow_missing,
                default_value=default_value,
            )
            return (parsed,)
        except ValueError as e:
            if allow_missing:
                # If allow_missing, shouldn't raise, but just in case
                return (template,)
            raise ValueError(f"Template parsing failed: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "required": {
                "pipe": ("PIPE",),
                "template": ("STRING", {"multiline": True, "default": ""}),
                "allow_missing": ("BOOLEAN", {"default": False}),
                "default_value": ("STRING", {"default": "[MISSING]"}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("STRING",)
    RETURN_NAMES: Tuple[str, ...] = ("parsed_text",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
