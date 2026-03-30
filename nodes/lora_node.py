"""
All-to-Pipe LoRA node.

Adds one LoRA specification to the Pipe with companion file loading and parameter adjustment.
"""

from typing import Dict, Any, Tuple, Optional, List, Set
import os
import random
from torch import Tensor
from ..alltopipe_types import (
    Pipe,
    LoraSpec,
    PositivePrompt,
    NegativePrompt,
    LoraProcessor,
    ModelProcessor,
)

# from ..common.utils import deep_copy_pipe
from ..common.constants import MIN_LORA_WEIGHT, MAX_LORA_WEIGHT
from ..common.file_helpers import discover_loras_in_subfolder
from ..common.companion_loader import CompanionLoader


def validate_lora_spec(lora: LoraSpec) -> None:
    """
    Validate a LoRA specification.

    Raises:
        ValueError: If LoRA spec is invalid
    """
    if lora.weight < MIN_LORA_WEIGHT or lora.weight > MAX_LORA_WEIGHT:
        raise ValueError(
            f"LoRA weight must be between {MIN_LORA_WEIGHT} and {MAX_LORA_WEIGHT}, "
            f"got {lora.weight}"
        )
    if lora.clip_weight < MIN_LORA_WEIGHT or lora.clip_weight > MAX_LORA_WEIGHT:
        raise ValueError(
            f"LoRA clip_weight must be between {MIN_LORA_WEIGHT} and {MAX_LORA_WEIGHT}, "
            f"got {lora.clip_weight}"
        )


class LoraNode:
    """
    Adds one LoRA specification to the Pipe with dynamic selection.

    Features:
    - COMBO selector for LoRA subfolders
    - COMBO selector for available LoRAs in subfolder
    - Random LoRA selection
    - Loads and applies companion JSON file data
    - Parses and distributes prompt data
    - Appends to Pipe.loras list (allows chaining multiple LoRA nodes)
    """

    def __init__(self) -> None:
        """Initialize the LoRA node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe |None = None,
        lora_selection: str = "",
        weight: float = 1.0,
        clip_weight: float = 1.0,
        load_companion: bool = False,
        random_subfolder: str = "all",
    ) -> tuple[Pipe]:
        """
        Execute the node and add a LoRA specification to the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            lora_selection: LoRA selection (either "RANDOM /" or "subfolder/lora_name.ext")
            weight: Model weight strength (overridden by companion file if present)
            clip_weight: CLIP weight strength (overridden by companion file if present)
            load_companion: Whether to load weights from companion file
            random_subfolder: Subfolder to randomly select from when "RANDOM /" is chosen

        Returns:
            Tuple containing the modified Pipe instance
        """
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()

        if not new_pipe.model:
            raise ValueError("Pipe Needs a model before applying loras")

        # Handle RANDOM selection
        if lora_selection == "RANDOM /":
            # Get LoRAs from specified subfolder or all subfolders
            if random_subfolder == "all":
                all_loras = LoraNode._get_all_loras()
            else:
                all_loras = discover_loras_in_subfolder(random_subfolder)

            if not all_loras:
                raise ValueError(
                    f"No LoRAs found in subfolder: {random_subfolder if random_subfolder != 'all' else 'any'}"
                )
            lora_selection = (
                random.choice(all_loras)
                if random_subfolder == "all"
                else f"{random_subfolder}/{random.choice(all_loras)}"
            )

        # Parse lora selection string (format: "subfolder/lora_name.ext" or "lora_name.ext")
        if "/" in lora_selection:
            parts = lora_selection.rsplit(
                "/", 1
            )  # Split from right to handle subfolders with /
            lora_subfolder = parts[0]
            lora_name = parts[1]
        else:
            lora_subfolder = ""
            lora_name = lora_selection

        # Get available LoRAs in subfolder for validation
        available_loras = discover_loras_in_subfolder(lora_subfolder)

        if not available_loras:
            raise ValueError(f"No LoRAs found in subfolder: {lora_subfolder}")

        if lora_name not in available_loras:
            raise ValueError(
                f"LoRA {lora_name} not found in {lora_subfolder or 'root folder'}"
            )

        # Load companion file if requested to get weights

        lora_spec: LoraSpec = LoraSpec(
            name=lora_name,
            subfolder=lora_subfolder,
            weight=weight,
            clip_weight=clip_weight,
        )
        lora_weights: Dict[str, Tensor] = LoraProcessor.load_lora(lora_spec)
        (model, _, _) = ModelProcessor.load_model(new_pipe.model)

        model_keys: Set[str] = LoraProcessor.get_model_key_set(model)
        if not LoraProcessor.is_lora_compatible(lora_weights, model_keys):
            print(f"Architecture Mismatch: Skipping {lora_spec.name}")
            return (new_pipe,)

        companion = (
            CompanionLoader.load_lora_companion(lora_name, lora_subfolder)
            if load_companion
            else None
        )

        final_weight = weight
        final_clip_weight = clip_weight

        if companion is not None and hasattr(companion, "raw_data"):
            # Try to load weights from companion file
            weight_data: List[float] = companion.raw_data.get("weight", [])
            if len(weight_data) > 0:
                if len(weight_data) == 1:
                    final_weight = final_clip_weight = float(weight_data[0])
                elif len(weight_data) == 2:
                    final_weight = random.uniform(min(weight_data), max(weight_data))
                else:
                    final_weight = random.choice(weight_data)
                final_clip_weight = final_weight

        # Clamp weights to valid range
        lora_spec.weight = max(MIN_LORA_WEIGHT, min(MAX_LORA_WEIGHT, final_weight))
        lora_spec.clip_weight = max(
            MIN_LORA_WEIGHT, min(MAX_LORA_WEIGHT, final_clip_weight)
        )

        # Create and validate the LoRA specification

        # validate_lora_spec(lora_spec)

        if companion is not None:
            # Store companion file data if not already stored (LoRA takes priority over model)
            if new_pipe.companion_lora_data is None:
                new_pipe.companion_lora_data = []
            new_pipe.companion_lora_data.append(companion.raw_data)

            if companion.positive_prompt:
                if new_pipe.positive_prompt is None:
                    new_pipe.positive_prompt = PositivePrompt()
                new_pipe.positive_prompt.lora = CompanionLoader.apply_text_suggestions(
                    companion.positive_prompt,
                    new_pipe.positive_prompt.lora,
                    "Positive Prompts",
                )
            if companion.negative_prompt:
                if new_pipe.negative_prompt is None:
                    new_pipe.negative_prompt = NegativePrompt()
                new_pipe.negative_prompt.lora = CompanionLoader.apply_text_suggestions(
                    companion.negative_prompt,
                    new_pipe.negative_prompt.lora,
                    "Negative Prompts",
                )

            if new_pipe.parameters:
                new_pipe.parameters = CompanionLoader.apply_companion_to_parameters(
                    companion, new_pipe.parameters
                )
            if companion.resolution:
                new_pipe.image_config = CompanionLoader.apply_companion_to_image_config(
                    companion, new_pipe.image_config
                )
            if companion.clip_skip:
                new_pipe.model = CompanionLoader.apply_companion_to_model(
                    companion, new_pipe.model
                )
        new_pipe.loras.append(lora_spec)

        return (new_pipe,)

    @staticmethod
    def _get_all_loras(base_path: str = "models/loras") -> List[str]:
        """
        Recursively discover all LoRAs in all subfolders.

        Args:
            base_path: Base path to loras directory

        Returns:
            List of LoRA paths in format "subfolder/lora_name.ext" or "lora_name.ext"
        """
        loras: List[str] = []
        lora_extensions: Tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth")

        if not os.path.isdir(base_path):
            return loras

        try:
            # Walk all subdirectories
            for root, _, files in os.walk(base_path):
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in lora_extensions):
                        # Get relative path from base
                        rel_path = os.path.relpath(
                            os.path.join(root, filename), base_path
                        )
                        # Normalize path separators to forward slash
                        rel_path = rel_path.replace(os.sep, "/")
                        loras.append(rel_path)
        except (OSError, PermissionError):
            pass

        loras.sort()
        return loras

    @staticmethod
    def _get_lora_subfolders(base_path: str = "models/loras") -> List[str]:
        """
        Get all available LoRA subfolders.

        Args:
            base_path: Base path to loras directory

        Returns:
            List of subfolder names
        """
        subfolders: List[str] = ["all"]  # Include "all" option

        if not os.path.isdir(base_path):
            return subfolders

        try:
            for item in os.listdir(base_path):
                path = os.path.join(base_path, item)
                if os.path.isdir(path):
                    subfolders.append(item)
        except (OSError, PermissionError):
            pass

        # subfolders.sort()
        return subfolders

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with improved selectors.

        Returns:
            Dictionary defining node inputs with all LoRAs discovered recursively
        """
        # Get all LoRAs recursively from all subfolders
        all_loras = cls._get_all_loras()

        # Get available subfolders for random selection
        lora_subfolders = cls._get_lora_subfolders()

        # Add RANDOM option
        lora_options = ["RANDOM /"] + all_loras if all_loras else ["RANDOM /"]
        default_lora = all_loras[0] if all_loras else "RANDOM /"

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "lora_selection": (
                    (lora_options,)
                    if lora_options
                    else ("STRING", {"default": default_lora})
                ),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "clip_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "load_companion": ("BOOLEAN", {"default": True}),
                "random_subfolder": (
                    (lora_subfolders,)
                    if lora_subfolders
                    else ("STRING", {"default": "all"})
                ),
            },
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
