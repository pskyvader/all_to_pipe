"""
All-to-Pipe export JSON node.

Exports Pipe as plain serializable data.
"""

import json
from typing import Any
from pathlib import Path


from ..alltopipe_types import (
    Pipe,
    Model,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
    Template,
    TemplateParser,
)

from ..common.prompt_helpers import prompt_to_string


class ExportJsonNode:
    """
    Exports Pipe as plain serializable data.

    Produces a complete JSON representation of the pipeline
    including all parameters, prompts, models, LoRAs, and metadata
    without loading models or encoding prompts.
    """

    def __init__(self) -> None:
        """Initialize the export JSON node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe,
        model: bool = True,
        loras: bool = True,
        parameters: bool = True,
        image_config: bool = True,
        prompt_text: bool = True,
        prompt_map: bool = True,
        companion_data: bool = True,
    ) -> tuple[str]:
        """
        Execute the node and export pipe as JSON-safe data.

        Args:
            pipe: The input Pipe instance

        Returns:
            tuple containing:
                - json_string: Complete JSON representation of the pipeline

        Raises:
            ValueError: If pipe is invalid
        """
        if not isinstance(pipe.model, Model):
            raise ValueError("Pipe.model must be a Model instance")
        if not isinstance(pipe.parameters, Parameters):
            raise ValueError("Pipe.parameters must be a Parameters instance")
        if not isinstance(pipe.image_config, ImageConfig):
            raise ValueError("Pipe.image_config must be an ImageConfig instance")
        if not isinstance(pipe.positive_prompt, PositivePrompt):
            raise ValueError("Pipe.positive_prompt must be a PositivePrompt instance")
        if not isinstance(pipe.negative_prompt, NegativePrompt):
            raise ValueError("Pipe.negative_prompt must be a NegativePrompt instance")
        if not isinstance(pipe.positive_template, Template):
            raise ValueError("Pipe.positive_template must exist")
        if not isinstance(pipe.negative_template, Template):
            raise ValueError("Pipe.negative_template must exist")

        # Assemble consolidated flat JSON structure
        # All properties at top level for easier consumption
        json_output: dict[str, Any] = {}

        if model:
            json_output["model"] = Path(pipe.model.name).stem
            json_output["model_subfolder"] = pipe.model.subfolder
            json_output["clip_skip"] = pipe.model.clip_skip

        if loras and pipe.loras:
            lora_data: list[dict[str, Any]] = [
                {
                    "name": Path(lora.name).stem,
                    "subfolder": lora.subfolder,
                    "weight": lora.weight,
                    "clip_weight": lora.clip_weight,
                }
                for lora in pipe.loras
            ]
            json_output["loras"] = lora_data
            json_output["lora"] = Path(lora_data[0]["name"]).stem
            json_output["lora_weight"] = lora_data[0]["weight"]

        if parameters:
            json_output["steps"] = pipe.parameters.steps
            json_output["cfg"] = pipe.parameters.cfg
            json_output["sampler"] = pipe.parameters.sampler
            json_output["scheduler"] = pipe.parameters.scheduler
            json_output["seed"] = pipe.parameters.seed
            json_output["denoise"] = pipe.parameters.denoise

        if image_config:

            json_output["width"] = pipe.image_config.width
            json_output["height"] = pipe.image_config.height
            json_output["batch_size"] = pipe.image_config.batch_size
            json_output["image_noise"] = pipe.image_config.noise
            if pipe.image_config.color_code is not None:
                json_output["color_code"] = pipe.image_config.color_code

        if prompt_text:
            if pipe.positive_template.parsed_template is None:
                pipe.positive_template.parsed_template = TemplateParser.parse_template(
                    pipe.positive_template.text,
                    pipe.positive_prompt,
                    pipe.positive_template.allow_missing,
                )
            json_output["positive_prompt"] = pipe.positive_template.parsed_template

            if pipe.negative_template.parsed_template is None:
                pipe.negative_template.parsed_template = TemplateParser.parse_template(
                    pipe.negative_template.text,
                    pipe.negative_prompt,
                    pipe.negative_template.allow_missing,
                )
            json_output["negative_prompt"] = pipe.negative_template.parsed_template

        if prompt_map:
            json_output["positive_prompt_map"] = prompt_to_string(pipe.positive_prompt)
            json_output["negative_prompt_map"] = prompt_to_string(pipe.negative_prompt)
        if companion_data:
            json_output["companion"] = {}
            if pipe.companion_model_data is not None:
                json_output["companion"]["model"] = pipe.companion_model_data
            if pipe.companion_lora_data is not None:
                json_output["companion"]["loras"] = pipe.companion_lora_data

        # Return as JSON string
        json_string: str = json.dumps(json_output, indent=2, default=str)
        return (json_string,)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            dictionary defining node inputs
        """
        return {
            "required": {
                "pipe": ("PIPE",),
            },
            "optional": {
                "model": ("BOOLEAN", {"default": True}),
                "loras": ("BOOLEAN", {"default": True}),
                "parameters": ("BOOLEAN", {"default": True}),
                "image_config": ("BOOLEAN", {"default": True}),
                "prompt_text": ("BOOLEAN", {"default": True}),
                "prompt_map": ("BOOLEAN", {"default": True}),
                "companion_data": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("STRING",)
    RETURN_NAMES: tuple[str, ...] = ("json",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
