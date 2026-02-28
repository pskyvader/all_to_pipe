"""
Model type and processor for All-to-Pipe.

Handles model references and loading from checkpoints.
"""

from typing import Optional, Any, Dict


class Model:
    """
    Represents a single checkpoint model reference.
    No loading happens here.
    """

    def __init__(self, name: str, subfolder: str) -> None:
        """
        Initialize Model reference.

        Args:
            name: Model filename
            subfolder: Subfolder within models directory
        """
        self.name: str = name
        self.subfolder: str = subfolder


class ModelProcessor:
    """
    Processor for Model operations.
    Handles loading models from checkpoints and integrating with ComfyUI.
    """

    @staticmethod
    def load_model(model: Model) -> Optional[Any]:
        """
        Load a model from checkpoint.

        Args:
            model: Model instance with name and subfolder

        Returns:
            Loaded MODEL object compatible with ComfyUI KSampler
            or None if loading fails

        Raises:
            ValueError: If model name is invalid
        """
        if not model or not model.name:
            raise ValueError("Model name is required and cannot be empty")

        # STUB: Would import from ComfyUI model loader
        # This is where actual ComfyUI model loading happens
        # e.g., from comfy_api.models import load_checkpoint
        # return load_checkpoint(model.name, model.subfolder)

        return None
