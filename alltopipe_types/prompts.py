"""
Prompt types and processor for All-to-Pipe.

Handles positive and negative prompt containers and template parsing.
"""

from typing import Optional, Any


class PositivePrompt:
    """
    Strongly-typed container for positive prompt data.
    Attributes are added explicitly as needed.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = (
        "characters",
        "age",
        "body",
        "race",
        "face",
        "hair",
        "clothes",
        "accessories",
        "location",
        "action",
        "pose",
        "camera",
        "lighting",
        "style",
        "color",
        "environment",
        "embeddings",
        "tags",
    )

    def __init__(self) -> None:
        """Initialize empty positive prompt container."""
        self.template: Optional[str] = None
        self.allow_missing: bool = True
        self.model: Optional[str] = None
        self.lora: Optional[str] =None


class NegativePrompt:
    """
    Strongly-typed container for negative prompt data.
    Kept separate from PositivePrompt by design.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = (
        "permanent",
        "embeddings",
        "style",
        "color",
        "tags",
    )

    def __init__(self) -> None:
        """Initialize empty negative prompt container."""
        self.template: Optional[str] = None
        self.allow_missing: bool = True
        self.model: Optional[str] = None
        self.lora: Optional[str] =None



class PromptProcessor:
    """
    Processor for Prompt operations.
    Handles template parsing and prompt encoding.
    """

    @staticmethod
    def encode_prompt(
        prompt_text: str,
        clip: Any,
    ) -> Optional[Any]:
        """
        Encode a prompt text to CONDITIONING using ComfyUI CLIP encoder.

        Args:
            prompt_text: Raw prompt string to encode
            clip: Loaded CLIP object from ComfyUI

        Returns:
            Encoded CONDITIONING object compatible with KSampler
            or None if no clip provided

        Raises:
            ValueError: If prompt_text is invalid
        """
        if not prompt_text:
            raise ValueError("Prompt text cannot be empty")

        if clip is None:
            raise ValueError("CLIP model cannot be None for prompt encoding")

        # Use ComfyUI CLIP encoder to encode the prompt
        # Standard approach: tokenize -> encode from tokens
        tokens = clip.tokenize(prompt_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return conditioning
