"""
All-to-Pipe model node.

Assigns a model to an existing Pipe with companion file loading and parameter adjustment.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import random
from ..alltopipe_types import Pipe, Model, PositivePrompt, NegativePrompt
from ..common.utils import deep_copy_pipe
from ..common.file_helpers import discover_model_subfolders, discover_models_in_subfolder
from ..common.companion_loader import CompanionLoader


class ModelNode:
    """
    Assigns a model to an existing Pipe with dynamic subfolder and model selection.
    
    Features:
    - COMBO selector for model subfolders
    - COMBO selector for available models in subfolder
    - Random model selection
    - Loads and applies companion JSON file limits
    - Parses and distributes prompt data
    """

    def __init__(self) -> None:
        """Initialize the model node."""
        pass
    #TODO: add subfolder for when RANDOM/ is selected
    @staticmethod
    def execute(
        pipe: Optional[Pipe] = None,
        model_selection: str = "",
        load_companion: bool = False,
    ) -> Tuple[Pipe]:
        """
        Execute the node and assign a model to the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            model_selection: Model selection (either "RANDOM /" or "subfolder/model_name.ext")
            load_companion: Whether to load and apply companion file data

        Returns:
            Tuple containing the modified Pipe instance
        """
        new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()

        # Handle RANDOM selection
        if model_selection == "RANDOM /":
            # Get all models and select randomly
            all_models = ModelNode._get_all_models()
            if not all_models:
                raise ValueError("No models found in any subfolder")
            model_selection = random.choice(all_models)

        # Parse model selection string (format: "subfolder/model_name.ext" or "model_name.ext")
        if "/" in model_selection:
            parts = model_selection.rsplit("/", 1)  # Split from right to handle subfolders with /
            model_subfolder = parts[0]
            model_name = parts[1]
        else:
            model_subfolder = ""
            model_name = model_selection

        # Get available models in subfolder for validation
        available_models = discover_models_in_subfolder(model_subfolder)
        
        if not available_models:
            raise ValueError(f"No models found in subfolder: {model_subfolder}")

        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not found in {model_subfolder or 'root folder'}")

        # Create and attach the model
        new_pipe.model = Model(name=model_name, subfolder=model_subfolder)

        # Load companion file if requested
        if load_companion:
            companion = CompanionLoader.load_model_companion(model_name, model_subfolder)
            
            if companion is not None:
                # Store companion file as dictionary for later reference
                new_pipe.companion_model_data = companion.raw_data

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
                    # Store as model feature
                    new_pipe.positive_prompt.model = ", ".join(selected_prompts)

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
                    # Store as model feature
                    new_pipe.negative_prompt.model = ", ".join(selected_prompts)

        return (new_pipe,)

    @staticmethod
    def _get_all_models(base_path: str = "models/checkpoints") -> List[str]:
        """
        Recursively discover all models in all subfolders.
        
        Args:
            base_path: Base path to checkpoints directory
            
        Returns:
            List of model paths in format "subfolder/model_name.ext" or "model_name.ext"
        """
        models: List[str] = []
        model_extensions: Tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth")
        
        if not os.path.isdir(base_path):
            return models
        
        try:
            # Walk all subdirectories
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in model_extensions):
                        # Get relative path from base
                        rel_path = os.path.relpath(os.path.join(root, filename), base_path)
                        # Normalize path separators to forward slash
                        rel_path = rel_path.replace(os.sep, "/")
                        models.append(rel_path)
        except (OSError, PermissionError):
            pass
        
        models.sort()
        return models

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with improved selectors.

        Returns:
            Dictionary defining node inputs with all models discovered recursively
        """
        # Get all models recursively from all subfolders
        all_models = cls._get_all_models()
        
        # Add RANDOM option
        model_options = ["RANDOM /"] + all_models if all_models else ["RANDOM /"]
        default_model = all_models[0] if all_models else "RANDOM /"

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "model_selection": (model_options,) if model_options else ("STRING", {"default": default_model}),
                "load_companion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

