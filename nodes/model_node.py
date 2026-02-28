"""
All-to-Pipe model node.

Assigns a model to an existing Pipe with companion file loading and parameter adjustment.
"""

from typing import Dict, Any, Tuple, Optional, List
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

    @staticmethod
    def execute(
        pipe: Pipe,
        model_subfolder: str,
        model_name: str,
        random_model: bool = False,
    ) -> Tuple[Pipe]:
        """
        Execute the node and assign a model to the pipe.

        Args:
            pipe: The input Pipe instance
            model_subfolder: Subfolder path for the model
            model_name: Name of the model file (or ignored if random_model is True)
            random_model: If True, randomly select from available models

        Returns:
            Tuple containing the modified Pipe instance
        """
        new_pipe: Pipe = deep_copy_pipe(pipe)

        # Get available models in subfolder
        available_models = discover_models_in_subfolder(model_subfolder)
        
        if not available_models:
            raise ValueError(f"No models found in subfolder: {model_subfolder}")

        # Select model name
        if random_model:
            selected_model_name = random.choice(available_models)
        else:
            if model_name not in available_models:
                raise ValueError(f"Model {model_name} not found in {model_subfolder}")
            selected_model_name = model_name

        # Create and attach the model
        new_pipe.model = Model(name=selected_model_name, subfolder=model_subfolder)

        # Load companion file and apply limits/parameters
        companion = CompanionLoader.load_model_companion(selected_model_name, model_subfolder)
        
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

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with dynamic selectors.

        Returns:
            Dictionary defining node inputs with dynamic COMBO lists
        """
        # Get available subfolders
        subfolders = discover_model_subfolders()
        default_subfolder = subfolders[0] if subfolders else ""

        # Get models for default subfolder
        models_in_subfolder = discover_models_in_subfolder(default_subfolder)
        default_model = models_in_subfolder[0] if models_in_subfolder else ""

        return {
            "required": {
                "pipe": ("PIPE",),
                "model_subfolder": (subfolders,) if subfolders else ("STRING", {"default": ""}),
                "model_name": (models_in_subfolder,) if models_in_subfolder else ("STRING", {"default": default_model}),
                "random_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

