"""
All-to-Pipe template node.

Assigns and validates template strings for dynamic prompt parsing.
"""

from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe, PositivePrompt, NegativePrompt
from ..common.utils import deep_copy_pipe
from ..common.prompt_template import TemplateParser


class TemplateNode:
    """
    Assigns and parses template strings for dynamic prompt generation.

    Templates use {variable} syntax to reference prompt attributes from
    PositivePrompt and NegativePrompt classes.

    Example: "A {age} {body} person wearing {clothes} in a {background}"

    Supports validation and parsing of templates against available prompt data.
    """

    def __init__(self) -> None:
        """Initialize the template node."""
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "template_type": (["positive", "negative"],),
                "template_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A person wearing <color> <clothes>",
                    },
                ),
                "allow_missing": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE", "STRING")
    RETURN_NAMES: Tuple[str, ...] = ("pipe", "parsed_template")
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

    def execute(
        self,
        template_type: str,
        template_text: str,
        allow_missing: bool,
        pipe: Optional[Pipe] = None,
    ) -> Tuple[Pipe, str]:
        """
        Execute the node and assign template to pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            template_type: "positive" or "negative"
            template_text: Template string with <variable> placeholders
            Example: "A <age> <body> wearing <clothes>"

        Returns:
            Tuple containing the modified Pipe instance

        Raises:
            ValueError: If template_type is invalid
        """
        new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()

        if template_type not in ["positive", "negative"]:
            raise ValueError("template_type must be 'positive' or 'negative'")

        # If template is empty, skip
        if not template_text or not template_text.strip():
            raise ValueError("template_text cannot be empty")

        # Validate template syntax (find placeholders)
        placeholders = TemplateParser.find_placeholders(template_text)

        if template_type == "positive":
            # Store template in positive prompt
            if new_pipe.positive_prompt is None:
                new_pipe.positive_prompt = PositivePrompt()

            # Store template as a custom attribute
            new_pipe.positive_prompt.template = template_text

            # Also store required variables for reference
            new_pipe.positive_prompt.template_variables = placeholders
            new_pipe.positive_prompt.allow_missing = allow_missing

            parsed_template = TemplateParser.parse_template(
                new_pipe.positive_prompt.template,
                new_pipe.positive_prompt,
                allow_missing=new_pipe.positive_prompt.allow_missing,
                # default_value="",
            )

        else:  # negative
            # Store template in negative prompt
            if new_pipe.negative_prompt is None:
                new_pipe.negative_prompt = NegativePrompt()

            # Store template as a custom attribute
            new_pipe.negative_prompt.template = template_text

            # Also store required variables for reference
            new_pipe.negative_prompt.template_variables = placeholders
            new_pipe.negative_prompt.allow_missing = allow_missing
            parsed_template = TemplateParser.parse_template(
                new_pipe.negative_prompt.template,
                new_pipe.negative_prompt,
                allow_missing=new_pipe.positive_prompt.allow_missing,
                # default_value="",
            )

        return (new_pipe, parsed_template)
