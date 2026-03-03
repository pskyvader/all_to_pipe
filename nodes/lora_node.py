"""
All-to-Pipe LoRA node.

Adds one LoRA specification to the Pipe with companion file loading and parameter adjustment.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
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
    
    #TODO: add subfolder for when RANDOM/ is selected
    @staticmethod
    def execute(
        pipe: Optional[Pipe] = None,
        lora_selection: str = "",
        weight: float = 1.0,
        clip_weight: float = 1.0,
        load_companion: bool = False,
    ) -> Tuple[Pipe]:
        """
        Execute the node and add a LoRA specification to the pipe.
        
        Args:
            pipe: Optional Pipe instance (creates new if None)
            lora_selection: LoRA selection (either "RANDOM /" or "subfolder/lora_name.ext")
            weight: Model weight strength (overridden by companion file if present)
            clip_weight: CLIP weight strength (overridden by companion file if present)
            load_companion: Whether to load weights from companion file

        Returns:
            Tuple containing the modified Pipe instance
        """
        new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()

        # Handle RANDOM selection
        if lora_selection == "RANDOM /":
            # Get all LoRAs and select randomly
            all_loras = LoraNode._get_all_loras()
            if not all_loras:
                raise ValueError("No LoRAs found in any subfolder")
            lora_selection = random.choice(all_loras)

        # Parse lora selection string (format: "subfolder/lora_name.ext" or "lora_name.ext")
        if "/" in lora_selection:
            parts = lora_selection.rsplit("/", 1)  # Split from right to handle subfolders with /
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
            raise ValueError(f"LoRA {lora_name} not found in {lora_subfolder or 'root folder'}")

        # Load companion file if requested to get weights
        final_weight = weight
        final_clip_weight = clip_weight
        
        if load_companion:
            companion = CompanionLoader.load_lora_companion(lora_name, lora_subfolder)
            
            if companion is not None and hasattr(companion, 'raw_data') and isinstance(companion.raw_data, dict):
                # Try to load weights from companion file
                weight_data = companion.raw_data.get("weight", [])
                if isinstance(weight_data, list) and len(weight_data) > 0:
                    # Use first weight for model weight
                    final_weight = float(weight_data[0])
                    # Use second weight for clip_weight if available, otherwise use same as model weight
                    if len(weight_data) > 1:
                        final_clip_weight = float(weight_data[1])
                    else:
                        final_clip_weight = final_weight

        # Clamp weights to valid range
        final_weight = max(MIN_LORA_WEIGHT, min(MAX_LORA_WEIGHT, final_weight))
        final_clip_weight = max(MIN_LORA_WEIGHT, min(MAX_LORA_WEIGHT, final_clip_weight))

        # Create and validate the LoRA specification
        lora_spec: LoraSpec = LoraSpec(
            name=lora_name,
            subfolder=lora_subfolder,
            weight=final_weight,
            clip_weight=final_clip_weight,
        )
        validate_lora_spec(lora_spec)

        # Load companion file for prompts if requested
        if load_companion:
            companion = CompanionLoader.load_lora_companion(lora_name, lora_subfolder)
            
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
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in lora_extensions):
                        # Get relative path from base
                        rel_path = os.path.relpath(os.path.join(root, filename), base_path)
                        # Normalize path separators to forward slash
                        rel_path = rel_path.replace(os.sep, "/")
                        loras.append(rel_path)
        except (OSError, PermissionError):
            pass
        
        loras.sort()
        return loras

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with improved selectors.

        Returns:
            Dictionary defining node inputs with all LoRAs discovered recursively
        """
        # Get all LoRAs recursively from all subfolders
        all_loras = cls._get_all_loras()
        
        # Add RANDOM option
        lora_options = ["RANDOM /"] + all_loras if all_loras else ["RANDOM /"]
        default_lora = all_loras[0] if all_loras else "RANDOM /"

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "lora_selection": (lora_options,) if lora_options else ("STRING", {"default": default_lora}),
                "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "load_companion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

