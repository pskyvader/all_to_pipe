"""
Prompt types and processor for All-to-Pipe.

Handles positive and negative prompt containers and template parsing.
"""

from typing import Optional, Dict, Any, Union


class PositivePrompt:
    """
    Strongly-typed container for positive prompt data.
    Attributes are added explicitly as needed.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = ("age", "body", "clothes", "background", "face", "lora", "model")

    def __init__(self) -> None:
        """Initialize empty positive prompt container."""
        self.age: Optional[str] = None
        self.body: Optional[str] = None
        self.clothes: Optional[str] = None
        self.background: Optional[str] = None
        self.face: Optional[str] = None
        self.lora: Optional[str] = None
        self.model: Optional[str] = None


class NegativePrompt:
    """
    Strongly-typed container for negative prompt data.
    Kept separate from PositivePrompt by design.
    Encoding is NOT performed here.
    """

    ALLOWED_FEATURES: tuple[str, ...] = ("static", "lora", "model")

    def __init__(self) -> None:
        """Initialize empty negative prompt container."""
        self.static: Optional[str] = None
        self.lora: Optional[str] = None
        self.model: Optional[str] = None


class PromptProcessor:
    """
    Processor for Prompt operations.
    Handles template parsing and prompt encoding.
    """

    @staticmethod
    def parse_template(
        template: str,
        positive_prompt: PositivePrompt,
        negative_prompt: Optional[NegativePrompt] = None,
    ) -> str:
        """
        Parse a template string with variable substitution from prompt objects.

        Template format: "A {age} {body} wearing {clothes} in a {background}"
        Variables are replaced with values from prompt attributes.

        Args:
            template: Template string with {variable} placeholders
            positive_prompt: PositivePrompt instance with data
            negative_prompt: Optional NegativePrompt instance

        Returns:
            Parsed string with variables substituted

        Raises:
            ValueError: If a required variable is missing
        """
        if not template:
            return ""

        result: str = template
        import re

        # Find all placeholders
        placeholders: list[str] = re.findall(r"\{([^}]+)\}", template)

        for placeholder in placeholders:
            # Try to get from positive prompt first
            if hasattr(positive_prompt, placeholder):
                value: Optional[str] = getattr(positive_prompt, placeholder)
            elif negative_prompt and hasattr(negative_prompt, placeholder):
                value = getattr(negative_prompt, placeholder)
            else:
                value = None

            if value is None:
                raise ValueError(f"Variable '{placeholder}' not found in prompts")

            result = result.replace(f"{{{placeholder}}}", str(value))

        return result

    @staticmethod
    def encode_prompt(
        prompt_text: str,
        clip: Optional[Any] = None,
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

        # STUB: Would use ComfyUI CLIP encoder
        # This is where actual prompt encoding happens
        # e.g., from comfy_api.clip import encode_prompt
        # return encode_prompt(clip, prompt_text)

        return None

    @staticmethod
    def shuffle_and_subset_prompts(
        prompts: list[str],
        count: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """
        Shuffle prompts and optionally select a subset.

        Args:
            prompts: List of prompt strings
            count: Number of prompts to select, or None to keep all
            seed: Random seed for reproducibility

        Returns:
            List of selected prompts

        Raises:
            ValueError: If inputs are invalid
        """
        if not prompts or not isinstance(prompts, list):
            raise ValueError("Prompts must be a non-empty list")

        import random

        if seed is not None:
            random.seed(seed)

        shuffled: list[str] = prompts.copy()
        random.shuffle(shuffled)

        if count is None:
            return shuffled

        if count <= 0:
            raise ValueError("Count must be positive or None")

        return shuffled[: min(count, len(shuffled))]
