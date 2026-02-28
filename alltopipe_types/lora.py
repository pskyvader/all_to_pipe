"""
LoRA type and processor for All-to-Pipe.

Handles LoRA specifications and application to models.
"""

from typing import Optional, Any, List


class LoraSpec:
    """
    Single LoRA entry.
    Multiple instances may exist in Pipe.loras.
    """

    def __init__(
        self,
        name: str,
        subfolder: str,
        weight: float,
        clip_weight: float,
    ) -> None:
        """
        Initialize LoRA specification.

        Args:
            name: LoRA filename
            subfolder: Subfolder within loras directory
            weight: Model weight strength
            clip_weight: CLIP weight strength
        """
        self.name: str = name
        self.subfolder: str = subfolder
        self.weight: float = weight
        self.clip_weight: float = clip_weight


class LoraProcessor:
    """
    Processor for LoRA operations.
    Handles applying LoRAs to loaded models.
    """

    @staticmethod
    def apply_lora(
        model: Any,
        clip: Any,
        loras: List[LoraSpec],
    ) -> tuple[Any, Any]:
        """
        Apply a list of LoRAs to a model and clip.

        Args:
            model: Loaded MODEL object from ComfyUI
            clip: Loaded CLIP object from ComfyUI
            loras: List of LoraSpec instances to apply

        Returns:
            Tuple of (modified_model, modified_clip) with LoRAs applied
            or (model, clip) unchanged if no LoRAs provided

        Raises:
            ValueError: If lora list is invalid
        """
        if not loras:
            return (model, clip)

        if model is None or clip is None:
            raise ValueError("Model and Clip cannot be None")

        # STUB: Would import from ComfyUI lora loader
        # This is where actual ComfyUI LoRA application happens
        # e.g., from comfy_api.loaders import load_lora
        # for lora_spec in loras:
        #     model, clip = load_lora(model, clip, lora_spec.name, lora_spec.weight, lora_spec.clip_weight)

        return (model, clip)
