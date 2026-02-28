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

class ExportNode:
    """
    Resolves Pipe into sampler-ready objects.
    
    - Loads model
    - Applies all LoRAs
    - Encodes prompts
    - Exports parameters ready to insert in a KSampler node
    - Returns: model, positive conditioning, negative conditioning, sampler parameters
    """

    def __init__(self) -> None:
        """Initialize the export node."""
        pass

    @staticmethod
    def execute(pipe: Pipe) -> Tuple[Any, ...]:
        """
        Execute the node and resolve pipe into sampler-ready objects.
        
        Args:
            pipe: The input Pipe instance (must be fully configured)
            
        Returns:
            Tuple containing:
                - model: Loaded MODEL object (or None if not loaded)
                - positive: Encoded CONDITIONING from positive prompt
                - negative: Encoded CONDITIONING from negative prompt
                - sampler_params: Dictionary of parameters for KSampler
                
        Raises:
            ValueError: If pipe is invalid or incomplete
        """
        # Validate the pipe has all required fields
        validate_pipe(pipe)

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

        # Parse and merge positive prompt
        positive_prompt_text: str = merge_prompts(
            {k: v for k, v in pipe.positive_prompt.__dict__.items() if v is not None}
        )
        positive_conditioning: Optional[Any] = None
        if positive_prompt_text:
            try:
                positive_conditioning = PromptProcessor.encode_prompt(
                    positive_prompt_text, clip
                )
            except ValueError:
                # Encoding failed, continue with None
                positive_conditioning = None

        # Parse and merge negative prompt
        negative_prompt_text: str = merge_prompts(
            {k: v for k, v in pipe.negative_prompt.__dict__.items() if v is not None}
        )
        negative_conditioning: Optional[Any] = None
        if negative_prompt_text:
            try:
                negative_conditioning = PromptProcessor.encode_prompt(
                    negative_prompt_text, clip
                )
            except ValueError:
                # Encoding failed, continue with None
                negative_conditioning = None

        # Build sampler parameters dictionary
        sampler_params: Dict[str, Any] = {
            "steps": pipe.parameters.steps,
            "cfg": pipe.parameters.cfg,
            "sampler_name": pipe.parameters.sampler,
            "scheduler": pipe.parameters.scheduler,
            "seed": pipe.parameters.seed,
        }       
        #todo: export all parameters as single parameter, so it can be connected to ksampler directly
        return (model, positive_conditioning, negative_conditioning, sampler_params)

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

    RETURN_TYPES: Tuple[str, ...] = ("MODEL", "CONDITIONING", "CONDITIONING", "BASIC_PIPE")
    RETURN_NAMES: Tuple[str, ...] = ("model", "positive", "negative", "sampler_params")
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
