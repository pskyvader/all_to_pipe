"""
All-to-Pipe export JSON node.

Exports Pipe as plain serializable data.
"""

import json
from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe
from ..common.prompt_helpers import prompt_to_string
from ..common.validators import validate_pipe


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

        #TODO: all properties should be combined in a single dictionary, 
        # not several sub dictionaries
        # Assemble complete JSON output with consolidated structure
        json_output: Dict[str, Any] = {
            "pipeline": {
                "model": model_data,
                "loras": loras_data,
                "parameters": parameters_data,
                "prompts": {
                    "positive": positive_dict,
                    "negative": negative_dict,
                },
                "image_config": image_config_data,
                "metadata": {
                    "companion": companion_data if companion_data else None,
                }
            }
        }

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
