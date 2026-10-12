"""
All-to-Pipe template node.

Assigns and validates template strings for dynamic prompt parsing.
"""

import json
from typing import Any
from ..alltopipe_types import (
    Pipe,
    # PositivePrompt,
    # NegativePrompt,
    Template,
    TemplateParser,
    PromptProcessor,
)

# from ..common.utils import deep_copy_pipe


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
    def INPUT_TYPES(cls) -> dict[str, Any]:
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
                "allow_parse_errors": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When on, prompt problems are repaired and logged as warnings: missing placeholders become empty text, and malformed emphasis like (:1.1) or unbalanced parentheses are stripped. When off, any of these raises an error instead.",
                    },
                ),
                "decay": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Strengthens a linear falloff applied to whole chunks, so each successive chunk fades relative to the one before it. Applies only when the prompt exceeds the model's encoding length. 0.0 is lossless and leaves every chunk at full weight; higher values fade later chunks toward the decay floor.",
                    },
                ),
                "decay_floor": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Lowest multiplier the decay can reach. Only takes effect when decay is above 0.0.",
                    },
                ),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE", "STRING")
    RETURN_NAMES: tuple[str, ...] = ("pipe", "parsed_template")
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

    def execute(
        self,
        template_type: str,
        template_text: str,
        allow_parse_errors: bool,
        decay: float,
        decay_floor: float,
        pipe: Pipe | None = None,
    ) -> tuple[Pipe, str, str]:
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
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()
        # Validate template syntax (find placeholders)
        placeholders = TemplateParser.find_placeholders(template_text)
        template = Template(
            template_type,
            placeholders,
            template_text,
            allow_parse_errors,
            decay,
            decay_floor,
        )

        if template_type == "positive":
            # Store template in positive prompt
            if new_pipe.positive_prompt is not None:
                template.parsed_template = TemplateParser.parse_template(
                    template_text,
                    new_pipe.positive_prompt,
                    allow_parse_errors,
                )
            new_pipe.positive_template = template

        else:  # negative
            # Store template in negative prompt
            if new_pipe.negative_prompt is not None:
                template.parsed_template = TemplateParser.parse_template(
                    template_text,
                    new_pipe.negative_prompt,
                    allow_parse_errors,
                )

            new_pipe.negative_template = template

        # Chunk breakdown of the template just assigned.
        chunk_report = self.describe_chunks(new_pipe, template)

        return (
            new_pipe,
            template.parsed_template if template.parsed_template else "",
            chunk_report,
        )

    @staticmethod
    def describe_chunks(pipe: Pipe, template: Template) -> str:
        """Reports how a template splits into chunks and what weights its tokens get.

        One line per chunk, with consecutive equal-weight tokens merged into a run,
        so a long prompt stays scannable:

            chunk 1/3  tokens 0-74   decay x1
              (a red cat,1*3); (blurry,0.5); (soft light,1*2)

        Reads the clip that ModelNode already cached on the pipe, so this costs no
        additional model load and runs no encode pass. Weights come from the
        template instance, so they match what ExportNode will encode.
        """
        if pipe.model is None:
            return json.dumps({"error": "no model set - connect a ModelNode upstream"})

        cached = getattr(pipe.model, "cached_model", None)
        clip = cached[1] if cached is not None else None
        if clip is None:
            return json.dumps({"error": "model set but clip not loaded yet"})

        if not template.parsed_template:
            return json.dumps({"error": "no prompt text to chunk"})

        described = PromptProcessor.describe_prompt(
            clip,
            template.parsed_template,
            template.decay,
            template.decay_floor,
        )

        lines: list[str] = []
        total: int = len(described)
        for index, (position, multiplier, summary) in enumerate(described):
            count: int = PromptProcessor.count_summary_tokens(summary)
            end: int = position + count - 1
            lines.append(
                f"chunk {index + 1}/{total}  tokens {position}-{end}  "
                f"({count})  decay x{multiplier:g}"
            )
            lines.append(f"  {summary}")
        return "\n".join(lines)

    RETURN_TYPES: tuple[str, ...] = ("PIPE", "STRING", "STRING")
    RETURN_NAMES: tuple[str, ...] = ("pipe", "parsed_template", "chunks")
