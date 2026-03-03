"""
All-to-Pipe negative prompt node.

Populates the negative prompt container.
"""

from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe, NegativePrompt
from ..common.utils import deep_copy_pipe


class NegativePromptNode:
    """
    Populates the negative prompt container.

    Modifies attributes on Pipe.negative_prompt.
    Encoding is NOT performed here.
    """

    def __init__(self) -> None:
        """Initialize the negative prompt node."""
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs with feature selector and text input
        """
        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "feature": (NegativePrompt.ALLOWED_FEATURES,),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

    def execute(self, pipe: Optional[Pipe] = None, feature: str = None, text: str = None) -> Tuple[Pipe]:
        """
        Execute the node and populate negative prompts.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            feature: Feature name from ALLOWED_FEATURES
            text: Negative prompt text

        Returns:
            Tuple containing the modified Pipe instance

        Raises:
            ValueError: If feature is not allowed
        """
        new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()

        if new_pipe.negative_prompt is None:
            new_pipe.negative_prompt = NegativePrompt()

        if feature not in NegativePrompt.ALLOWED_FEATURES:
            raise ValueError(f"Invalid feature: {feature}")

        setattr(new_pipe.negative_prompt, feature, text)

        return (new_pipe,)
