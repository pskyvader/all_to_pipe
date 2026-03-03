"""
All-to-Pipe export JSON node.

Exports Pipe as plain serializable data.
"""

import json
from typing import Dict, Any, Tuple, Optional
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
    def execute(pipe: Pipe) -> Tuple[str,]:
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

        # Convert prompts to dictionaries
        positive_dict: Dict[str, str] = prompt_to_string(pipe.positive_prompt)
        negative_dict: Dict[str, str] = prompt_to_string(pipe.negative_prompt)
        
        # Parse templates if they exist
        positive_template_parsed = None
        if hasattr(pipe.positive_prompt, "template") and pipe.positive_prompt.template:
            try:
                positive_template_parsed = TemplateParser.parse_template(
                    pipe.positive_prompt.template,
                    pipe.positive_prompt,
                    pipe.negative_prompt,
                    allow_missing=True,
                    default_value=""
                )
            except ValueError:
                pass
        
        negative_template_parsed = None
        if hasattr(pipe.negative_prompt, "template") and pipe.negative_prompt.template:
            try:
                negative_template_parsed = TemplateParser.parse_template(
                    pipe.negative_prompt.template,
                    pipe.positive_prompt,
                    pipe.negative_prompt,
                    allow_missing=True,
                    default_value=""
                )
            except ValueError:
                pass

        # Build LoRAs data
        loras_data: list[Dict[str, Any]] = []
        for lora in pipe.loras:
            loras_data.append(
                {
                    "name": lora.name,
                    "subfolder": lora.subfolder,
                    "weight": lora.weight,
                    "clip_weight": lora.clip_weight,
                }
            )

        # Build model data
        model_data: Optional[Dict[str, str]] = None
        if pipe.model is not None:
            model_data = {
                "name": pipe.model.name,
                "subfolder": pipe.model.subfolder,
            }

        # Build parameters data
        parameters_data: Optional[Dict[str, Any]] = None
        if pipe.parameters is not None:
            parameters_data = {
                "steps": pipe.parameters.steps,
                "cfg": pipe.parameters.cfg,
                "sampler": pipe.parameters.sampler,
                "scheduler": pipe.parameters.scheduler,
                "seed": pipe.parameters.seed,
            }

        # Build image config data
        image_config_data: Optional[Dict[str, Any]] = None
        if pipe.image_config is not None:
            image_config_data = {
                "width": pipe.image_config.width,
                "height": pipe.image_config.height,
                "batch_size": pipe.image_config.batch_size,
                "noise": pipe.image_config.noise,
                "color_code": pipe.image_config.color_code,
            }

        # Build companion file data if available
        companion_data: Dict[str, Any] = {}
        if pipe.companion_model_data is not None:
            companion_data["model"] = pipe.companion_model_data
        if pipe.companion_lora_data is not None:
            companion_data["loras"] = pipe.companion_lora_data

        # Assemble consolidated flat JSON structure
        # All properties at top level for easier consumption
        json_output: Dict[str, Any] = {}
        
        # Model info
        if model_data:
            json_output["model"] = model_data["name"]
            json_output["model_subfolder"] = model_data["subfolder"]
        
        # LoRAs
        if loras_data:
            json_output["loras"] = loras_data
        
        # Parameters
        if parameters_data:
            json_output["steps"] = parameters_data["steps"]
            json_output["cfg"] = parameters_data["cfg"]
            json_output["sampler"] = parameters_data["sampler"]
            json_output["scheduler"] = parameters_data["scheduler"]
            json_output["seed"] = parameters_data["seed"]
        
        # Prompts (flat structure with parsed templates if available)
        if positive_template_parsed:
            json_output["positive_prompt"] = positive_template_parsed
        elif positive_dict:
            json_output["positive_prompt"] = positive_dict
        
        if negative_template_parsed:
            json_output["negative_prompt"] = negative_template_parsed
        elif negative_dict:
            json_output["negative_prompt"] = negative_dict
        
        # Image config
        if image_config_data:
            json_output["width"] = image_config_data["width"]
            json_output["height"] = image_config_data["height"]
            json_output["batch_size"] = image_config_data["batch_size"]
            json_output["noise"] = image_config_data["noise"]
            if image_config_data["color_code"]:
                json_output["color_code"] = image_config_data["color_code"]
        
        # Companion data
        if companion_data:
            json_output["metadata"] = {"companion": companion_data}

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
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("STRING",)
    RETURN_NAMES: Tuple[str, ...] = ("json",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
