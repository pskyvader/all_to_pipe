"""
All-to-Pipe export node.

Resolves Pipe into sampler-ready objects.
"""

from typing import Dict, Any, Tuple, Optional
import torch
from ..alltopipe_types import Pipe
from ..alltopipe_types.model import ModelProcessor
from ..alltopipe_types.lora import LoraProcessor

from ..alltopipe_types.image_config import ImageConfigProcessor
from ..alltopipe_types.prompts import PromptProcessor
from ..common.validators import validate_pipe
from ..common.prompt_helpers import merge_prompts
from ..common.prompt_template import TemplateParser


from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES


class ExportNode:
    """
    Resolves Pipe into sampler-ready objects with KSampler-compatible outputs.

    - Loads model
    - Applies all LoRAs
    - Encodes prompts with template parsing support
    - Exports all parameters individually for direct KSampler connection
    - Creates empty LATENT image from image config dimensions
    - Returns: model, positive conditioning, negative conditioning, 6 sampling parameters, and latent

    All outputs connect directly to standard ComfyUI KSampler node:
    - model → KSampler.model
    - positive → KSampler.positive
    - negative → KSampler.negative
    - seed → KSampler.seed
    - steps → KSampler.steps
    - cfg → KSampler.cfg
    - sampler → KSampler.sampler_name
    - scheduler → KSampler.scheduler
    - denoise → KSampler.denoise
    - latent_image → KSampler.latent_image
    """

    def __init__(self) -> None:
        """Initialize the export node."""
        pass

    @staticmethod
    def execute(pipe: Pipe) -> Tuple[Any, ...]:
        """
        Execute the node and resolve pipe into sampler-ready objects.

        All parameters are exported individually for direct connection to KSampler nodes.
        Creates an empty LATENT image from image configuration dimensions.

        Args:
            pipe: The input Pipe instance (must be fully configured)

        Returns:
            Tuple containing 10 individual values:
                - model: Loaded MODEL object (or None)
                - positive: Encoded CONDITIONING from positive prompt
                - negative: Encoded CONDITIONING from negative prompt
                - seed: Random seed (INT)
                - steps: Number of sampling steps (INT)
                - cfg: Classifier-free guidance scale (FLOAT)
                - sampler: Sampler algorithm name (STRING)
                - scheduler: Scheduler algorithm name (STRING)
                - denoise: Denoise strength (FLOAT, range 0.0-1.0)
                - latent_image: Empty LATENT image tensor (created from image_config dimensions)

        Raises:
            ValueError: If pipe is invalid or incomplete
        """
        # Validate the pipe has all required fields
        validate_pipe(pipe)
        (model, clip, vae) = ModelProcessor.load_model(pipe.model)

        # Apply LoRAs if model was loaded and LoRAs are specified
        if pipe.loras:
            model, clip = LoraProcessor.apply_lora(model, clip, pipe.loras)

        # Parse and merge positive prompt (with template support)
        positive_prompt_dict = {
            k: v
            for k, v in pipe.positive_prompt.__dict__.items()
            if v is not None and k != "template" and k != "template_variables"
        }

        
        parsed_template = TemplateParser.parse_template(
            pipe.positive_prompt.template,
            pipe.positive_prompt,
            allow_missing=True,
            default_value="",
        )
        if parsed_template:
            positive_prompt_dict["template"] = parsed_template
    

        positive_prompt_text: str = merge_prompts(positive_prompt_dict)
        positive_conditioning: Optional[Any] = None
        if positive_prompt_text:
            positive_conditioning = PromptProcessor.encode_prompt(
                positive_prompt_text, clip
            )
        # Parse and merge negative prompt (with template support)
        negative_prompt_dict = {
            k: v
            for k, v in pipe.negative_prompt.__dict__.items()
            if v is not None and k != "template" and k != "template_variables"
        }

        parsed_template = TemplateParser.parse_template(
            pipe.negative_prompt.template,
            pipe.negative_prompt,
            allow_missing=True,
            default_value="",
        )
        if parsed_template:
            negative_prompt_dict["template"] = parsed_template
    

        negative_prompt_text: str = merge_prompts(negative_prompt_dict)
        negative_conditioning: Optional[Any] = None
        if negative_prompt_text:
            negative_conditioning = PromptProcessor.encode_prompt(
                negative_prompt_text, clip
            )
            
        # Extract all parameters for direct KSampler connection
        seed: int = pipe.parameters.seed
        steps: int = pipe.parameters.steps
        cfg: float = pipe.parameters.cfg
        sampler_name: str = pipe.parameters.sampler
        scheduler: str = pipe.parameters.scheduler

        denoise: float = 1.0  # Default denoise value

        latent_image=ImageConfigProcessor.create_noisy_latent(pipe.image_config, seed)
        

        # Return all parameters individually for direct KSampler connection
        # This allows connecting each output directly to KSampler inputs
        return (
            model,
            vae,
            positive_conditioning,
            negative_conditioning,
            latent_image,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        )

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

    RETURN_TYPES: Tuple[Any, ...] = (
        "MODEL",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "INT",
        "FLOAT",
        SAMPLER_NAMES,
        SCHEDULER_NAMES,
        "FLOAT",
    )
    RETURN_NAMES: Tuple[str, ...] = (
        "model",
        "vae",
        "positive",
        "negative",
        "latent_image",
        "seed",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "denoise",
    )
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
