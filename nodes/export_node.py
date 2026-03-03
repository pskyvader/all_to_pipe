"""
All-to-Pipe export node.

Resolves Pipe into sampler-ready objects.
"""

from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe
from ..alltopipe_types.model import ModelProcessor
from ..alltopipe_types.lora import LoraProcessor
from ..alltopipe_types.image_config import ImageConfigProcessor
from ..alltopipe_types.prompts import PromptProcessor
from ..common.validators import validate_pipe
from ..common.prompt_helpers import merge_prompts
from ..common.prompt_template import TemplateParser


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


#TODO: Sampler and scheduler, ksampler node expects a combo, but this node output is a string. correct that.

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

        # Import torch lazily (only when executing in ComfyUI context)
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "torch is required for ExportNode. This node must be run in ComfyUI context."
            )

        # Load model if specified
        model: Optional[Any] = None
        clip: Optional[Any] = None
        if pipe.model is not None:
            try:
                model = ModelProcessor.load_model(pipe.model)
                # Note: In real ComfyUI integration, clip would be extracted from model loading
                # For now, it remains None (stub)
            except ValueError:
                # Model loading failed, but continue with None
                model = None

        # Apply LoRAs if model was loaded and LoRAs are specified
        if model is not None and pipe.loras:
            try:
                # Note: LoraProcessor.apply_lora expects model, clip, and list of loras
                # In stub implementation, it returns (model, clip) unchanged
                model, clip = LoraProcessor.apply_lora(model, clip, pipe.loras)
            except ValueError:
                # LoRA application failed, continue with current model
                pass

        # Parse and merge positive prompt (with template support)
        positive_prompt_dict = {
            k: v
            for k, v in pipe.positive_prompt.__dict__.items()
            if v is not None and k != "template" and k != "template_variables"
        }

        # If template exists, parse it and add to prompt dict
        if hasattr(pipe.positive_prompt, "template") and pipe.positive_prompt.template:
            try:
                parsed_template = TemplateParser.parse_template(
                    pipe.positive_prompt.template,
                    pipe.positive_prompt,
                    pipe.negative_prompt,
                    allow_missing=True,
                    default_value="",
                )
                if parsed_template:
                    positive_prompt_dict["template"] = parsed_template
            except ValueError:
                # Template parsing failed, continue without it
                pass

        positive_prompt_text: str = merge_prompts(positive_prompt_dict)
        positive_conditioning: Optional[Any] = None
        if positive_prompt_text:
            try:
                positive_conditioning = PromptProcessor.encode_prompt(
                    positive_prompt_text, clip
                )
            except ValueError:
                # Encoding failed, continue with None
                positive_conditioning = None

        # Parse and merge negative prompt (with template support)
        negative_prompt_dict = {
            k: v
            for k, v in pipe.negative_prompt.__dict__.items()
            if v is not None and k != "template" and k != "template_variables"
        }

        # If template exists, parse it and add to prompt dict
        if hasattr(pipe.negative_prompt, "template") and pipe.negative_prompt.template:
            try:
                parsed_template = TemplateParser.parse_template(
                    pipe.negative_prompt.template,
                    pipe.positive_prompt,
                    pipe.negative_prompt,
                    allow_missing=True,
                    default_value="",
                )
                if parsed_template:
                    negative_prompt_dict["template"] = parsed_template
            except ValueError:
                # Template parsing failed, continue without it
                pass

        negative_prompt_text: str = merge_prompts(negative_prompt_dict)
        negative_conditioning: Optional[Any] = None
        if negative_prompt_text:
            try:
                negative_conditioning = PromptProcessor.encode_prompt(
                    negative_prompt_text, clip
                )
            except ValueError:
                # Encoding failed, continue with None
                negative_conditioning = None

        # Extract all parameters for direct KSampler connection
        seed: int = pipe.parameters.seed if pipe.parameters else 0
        steps: int = pipe.parameters.steps if pipe.parameters else 20
        cfg: float = pipe.parameters.cfg if pipe.parameters else 7.0
        sampler_name: str = pipe.parameters.sampler if pipe.parameters else "euler"
        scheduler: str = pipe.parameters.scheduler if pipe.parameters else "normal"

        # Extract image config if available
        width: int = pipe.image_config.width if pipe.image_config else 512
        height: int = pipe.image_config.height if pipe.image_config else 512
        batch_size: int = pipe.image_config.batch_size if pipe.image_config else 1
        denoise: float = 1.0  # Default denoise value

        # Create empty LATENT image from image config
        # LATENT format: dict with 'samples' key containing torch tensor
        # Shape: (batch_size, 4, height//8, width//8) - latent space is 8x smaller
        latent_height = height // 8
        latent_width = width // 8
        latent_image = {
            "samples": torch.zeros((batch_size, 4, latent_height, latent_width))
        }

        # Return all parameters individually for direct KSampler connection
        # This allows connecting each output directly to KSampler inputs
        return (
            model,
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

    RETURN_TYPES: Tuple[str, ...] = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "FLOAT",
    )
    RETURN_NAMES: Tuple[str, ...] = (
        "model",
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
