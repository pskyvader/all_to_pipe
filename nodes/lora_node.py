"""
All-to-Pipe LoRA node.

Adds one LoRA specification to the Pipe with companion file loading and parameter adjustment.
"""

from typing import Dict, Any, Tuple, Optional, List
import random
from ..alltopipe_types import Pipe, LoraSpec, PositivePrompt, NegativePrompt
from ..common.utils import deep_copy_pipe
from ..common.constants import MIN_LORA_WEIGHT, MAX_LORA_WEIGHT
from ..common.file_helpers import discover_lora_subfolders, discover_loras_in_subfolder
from ..common.companion_loader import CompanionLoader


def validate_lora_spec(lora: LoraSpec) -> None:
    """
    Validate a LoRA specification.
    
    Raises:
        ValueError: If LoRA spec is invalid
    """
    if lora is None:
        raise ValueError("LoRA spec cannot be None")
    if not isinstance(lora.name, str) or not lora.name:
        raise ValueError("LoRA name must be a non-empty string")
    if not isinstance(lora.subfolder, str):
        raise ValueError("LoRA subfolder must be a string")
    if not isinstance(lora.weight, (int, float)):
        raise ValueError(f"LoRA weight must be a number, got {type(lora.weight)}")
    if not isinstance(lora.clip_weight, (int, float)):
        raise ValueError(f"LoRA clip_weight must be a number, got {type(lora.clip_weight)}")
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
        pipe: Pipe,
        lora_subfolder: str,
        lora_name: str,
        weight: float = 1.0,
        clip_weight: float = 1.0,
        random_lora: bool = False,
    ) -> Tuple[Pipe]:
        """
        Execute the node and add a LoRA specification to the pipe.
        
        Args:
            pipe: The input Pipe instance
            lora_subfolder: Subfolder path for the LoRA
            lora_name: Name of the LoRA file (or ignored if random_lora is True)
            weight: Model weight strength
            clip_weight: CLIP weight strength
            random_lora: If True, randomly select from available LoRAs

        Returns:
            Tuple containing the modified Pipe instance
        """
        new_pipe: Pipe = deep_copy_pipe(pipe)

        # Get available LoRAs in subfolder
        available_loras = discover_loras_in_subfolder(lora_subfolder)
        
        if not available_loras:
            raise ValueError(f"No LoRAs found in subfolder: {lora_subfolder}")

        # Select LoRA name
        if random_lora:
            selected_lora_name = random.choice(available_loras)
        else:
            if lora_name not in available_loras:
                raise ValueError(f"LoRA {lora_name} not found in {lora_subfolder}")
            selected_lora_name = lora_name

        # Create and validate the LoRA specification
        lora_spec: LoraSpec = LoraSpec(
            name=selected_lora_name,
            subfolder=lora_subfolder,
            weight=weight,
            clip_weight=clip_weight,
        )
        validate_lora_spec(lora_spec)

        # Load companion file and apply limits/parameters
        companion = CompanionLoader.load_lora_companion(selected_lora_name, lora_subfolder)
        
        if companion is not None:
            # Store companion file data if not already stored (LoRA takes priority over model)
            if new_pipe.companion_lora_data is None:
                new_pipe.companion_lora_data = []
            new_pipe.companion_lora_data.append(companion.raw_data)

            # Extract and apply positive prompts if present
            if companion.positive_prompt:
                positive_prompts = companion.positive_prompt
                # Shuffle and select a random subset
                shuffled = list(positive_prompts)
                random.shuffle(shuffled)
                subset_size = random.randint(1, len(shuffled))
                selected_prompts = shuffled[:subset_size]
                
                if new_pipe.positive_prompt is None:
                    new_pipe.positive_prompt = PositivePrompt()
                # Append to lora feature (could exist from model)
                existing = getattr(new_pipe.positive_prompt, 'lora', None) or ""
                if existing:
                    new_pipe.positive_prompt.lora = existing + ", " + ", ".join(selected_prompts)
                else:
                    new_pipe.positive_prompt.lora = ", ".join(selected_prompts)

            # Extract and apply negative prompts if present
            if companion.negative_prompt:
                negative_prompts = companion.negative_prompt
                # Shuffle and select a random subset
                shuffled = list(negative_prompts)
                random.shuffle(shuffled)
                subset_size = random.randint(1, len(shuffled))
                selected_prompts = shuffled[:subset_size]
                
                if new_pipe.negative_prompt is None:
                    new_pipe.negative_prompt = NegativePrompt()
                # Append to lora feature (could exist from model)
                existing = getattr(new_pipe.negative_prompt, 'lora', None) or ""
                if existing:
                    new_pipe.negative_prompt.lora = existing + ", " + ", ".join(selected_prompts)
                else:
                    new_pipe.negative_prompt.lora = ", ".join(selected_prompts)

        # Append to the loras list
        new_pipe.loras.append(lora_spec)

        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with dynamic selectors.

        Returns:
            Dictionary defining node inputs with dynamic COMBO lists
        """
        # Get available subfolders
        subfolders = discover_lora_subfolders()
        default_subfolder = subfolders[0] if subfolders else ""

        # Get LoRAs for default subfolder
        loras_in_subfolder = discover_loras_in_subfolder(default_subfolder)
        default_lora = loras_in_subfolder[0] if loras_in_subfolder else ""

        return {
            "required": {
                "pipe": ("PIPE",),
                "lora_subfolder": (subfolders,) if subfolders else ("STRING", {"default": ""}),
                "lora_name": (loras_in_subfolder,) if loras_in_subfolder else ("STRING", {"default": default_lora}),
                "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "random_lora": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

