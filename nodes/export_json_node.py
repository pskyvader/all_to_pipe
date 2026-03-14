"""
All-to-Pipe export JSON node.

Exports Pipe as plain serializable data.
"""

import json
from typing import Dict, Any, Tuple, Optional, List
from ..alltopipe_types import Pipe
from ..common.prompt_helpers import prompt_to_string
from ..common.validators import validate_pipe
from ..common.prompt_template import TemplateParser


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
    ) -> Tuple[str,]:
        """
        Execute the node and export pipe as JSON-safe data.

        Args:
            pipe: The input Pipe instance

        Returns:
            Tuple containing:
                - json_string: Complete JSON representation of the pipeline

        Raises:
            ValueError: If pipe is invalid
        """
        # Validate the pipe
        validate_pipe(pipe)

        # Assemble consolidated flat JSON structure
        # All properties at top level for easier consumption
        json_output: Dict[str, Any] = {}

        if model:
            if pipe.model is not None:
                json_output["model"] = pipe.model.name
                json_output["model_subfolder"] = pipe.model.subfolder
                json_output["clip_skip"] = pipe.model.clip_skip

        if loras:
            lora_data: List[Dict[str, Any]] = [
                {
                    "name": lora.name,
                    "subfolder": lora.subfolder,
                    "weight": lora.weight,
                    "clip_weight": lora.clip_weight,
                }
                for lora in pipe.loras
            ]
            json_output["loras"] = lora_data
            json_output["lora"] = lora_data[0]["name"]
            json_output["lora_weight"] = lora_data[0]["weight"]

        if parameters:
            if pipe.parameters is not None:
                json_output["steps"] = pipe.parameters.steps
                json_output["cfg"] = pipe.parameters.cfg
                json_output["sampler"] = pipe.parameters.sampler
                json_output["scheduler"] = pipe.parameters.scheduler
                json_output["seed"] = pipe.parameters.seed
                json_output["denoise"] = pipe.parameters.denoise

        if image_config:
            if pipe.image_config is not None:

                json_output["width"] = pipe.image_config.width
                json_output["height"] = pipe.image_config.height
                json_output["batch_size"] = pipe.image_config.batch_size
                json_output["image_noise"] = pipe.image_config.noise
                if pipe.image_config.color_code is not None:
                    json_output["color_code"] = pipe.image_config.color_code

        if prompt_text:
            if pipe.positive_prompt is not None:
                if (
                    hasattr(pipe.positive_prompt, "template")
                    and pipe.positive_prompt.template
                ):
                    json_output["positive_prompt"] = TemplateParser.parse_template(
                        pipe.positive_prompt.template,
                        pipe.positive_prompt,
                        allow_missing=pipe.positive_prompt.allow_missing,
                    )

            if pipe.negative_prompt is not None:
                if (
                    hasattr(pipe.negative_prompt, "template")
                    and pipe.negative_prompt.template
                ):
                    json_output["negative_prompt"] = TemplateParser.parse_template(
                        pipe.negative_prompt.template,
                        pipe.negative_prompt,
                        allow_missing=pipe.negative_prompt.allow_missing,
                    )

        if prompt_map:
            if pipe.positive_prompt is not None:
                json_output["positive_prompt_map"] = prompt_to_string(
                    pipe.positive_prompt
                )
            if pipe.negative_prompt is not None:
                json_output["negative_prompt_map"] = prompt_to_string(
                    pipe.negative_prompt
                )
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
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
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

    RETURN_TYPES: Tuple[str, ...] = ("STRING",)
    RETURN_NAMES: Tuple[str, ...] = ("json",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
