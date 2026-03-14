"""
All-to-Pipe export node.

Resolves Pipe into sampler-ready objects.
"""

from typing import Dict, Any, Tuple
from ..alltopipe_types import Pipe
from ..alltopipe_types.model import ModelProcessor
from ..alltopipe_types.lora import LoraProcessor

from ..alltopipe_types.image_config import ImageConfigProcessor
from ..alltopipe_types.prompts import PromptProcessor
from ..common.validators import validate_pipe
from ..common.prompt_template import TemplateParser


from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
import gc
import torch
import comfy.model_management


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

        parsed_template = TemplateParser.parse_template(
            pipe.positive_prompt.template,
            pipe.positive_prompt,
            allow_missing=pipe.positive_prompt.allow_missing,
        )
        positive_conditioning = PromptProcessor.encode_prompt(parsed_template, clip)

        parsed_template = TemplateParser.parse_template(
            pipe.negative_prompt.template,
            pipe.negative_prompt,
            allow_missing=pipe.positive_prompt.allow_missing,
        )
        negative_conditioning = PromptProcessor.encode_prompt(parsed_template, clip)

        # Extract all parameters for direct KSampler connection
        seed: int = pipe.parameters.seed
        steps: int = pipe.parameters.steps
        cfg: float = pipe.parameters.cfg
        sampler_name: str = pipe.parameters.sampler
        scheduler: str = pipe.parameters.scheduler
        denoise: float = pipe.parameters.denoise

        if not isinstance(pipe.image_config.image, torch.Tensor):
            pipe.image_config.image = ImageConfigProcessor.create_noisy_image(
                pipe.image_config, seed
            )
        image = pipe.image_config.image

        latent_image: Dict[str, torch.Tensor] = {
            "samples": vae.encode(image[:, :, :, :3])
        }

        # since this node does all at the same time, it's important to clean up
        # the pipe
        del pipe
        gc.collect()
        comfy.model_management.soft_empty_cache()

        # Return all parameters individually for direct KSampler connection
        # This allows connecting each output directly to KSampler inputs
        return (
            model,
            vae,
            clip,
            positive_conditioning,
            negative_conditioning,
            image,
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
        "CLIP",
        "CONDITIONING",
        "CONDITIONING",
        "IMAGE",
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
        "clip",
        "positive",
        "negative",
        "image",
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
