"""
All-to-Pipe validators module.

Cross-node validation functions.
"""

from typing import Any

import torch

from ..alltopipe_types import (
    Pipe,
    Model,
    LoraSpec,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
)


def validate_pipe(pipe: Pipe) -> None:
    """
    Validate that a Pipe contains all required fields.

    This is a cross-node validator used by export nodes.

    Raises:
        ValueError: If required fields are missing or invalid

    Args:
        pipe: The Pipe instance to validate
    """
    if not isinstance(pipe.model, Model):
        raise ValueError("Pipe.model must be a Model instance")
    if not isinstance(pipe.parameters, Parameters):
        raise ValueError("Pipe.parameters must be a Parameters instance")
    if not isinstance(pipe.image_config, ImageConfig):
        raise ValueError("Pipe.image_config must be an ImageConfig instance")
    if not isinstance(pipe.positive_prompt, PositivePrompt):
        raise ValueError("Pipe.positive_prompt must be a PositivePrompt instance")
    if not isinstance(pipe.negative_prompt, NegativePrompt):
        raise ValueError("Pipe.negative_prompt must be a NegativePrompt instance")
    if not pipe.positive_prompt.template:
        raise ValueError("Pipe.positive_prompt.template must be a string")
    if not pipe.negative_prompt.template:
        raise ValueError("Pipe.negative_prompt.template must be a string")


def _wrong_type(key: str, expected: str, value: Any) -> ValueError:
    """Builds the error for a value that is not of the expected type at all."""
    return ValueError(
        f"'{key}' must be {expected}, got {type(value).__name__}: {value!r}"
    )


def _listed(choices: list[str] | tuple[str, ...], limit: int = 6) -> str:
    """Renders the first few choices, since the sampler list runs to dozens."""
    preview = ", ".join(str(choice) for choice in choices[:limit])
    if len(choices) > limit:
        preview += f", ... ({len(choices)} in total)"
    return preview


def validate_int(value: Any, key: str, minimum: int, maximum: int) -> int:
    """
    Check an integer field and return it unchanged.

    bool is rejected even though it is an int in Python: a switch wired into a
    count is a mistake, not a 0 or a 1.

    Raises:
        ValueError: If the value is not an INT inside the range
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _wrong_type(key, f"an INT between {minimum} and {maximum}", value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"'{key}' must be between {minimum} and {maximum}, got {value}"
        )
    return value


def validate_float(value: Any, key: str, minimum: float, maximum: float) -> float:
    """
    Check a numeric field and return it as a float.

    An INT is accepted and widened, so a whole-number socket can drive a
    float field such as cfg.

    Raises:
        ValueError: If the value is not a number inside the range
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _wrong_type(key, f"a FLOAT between {minimum} and {maximum}", value)

    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"'{key}' must be between {minimum} and {maximum}, got {number}"
        )
    return number


def validate_bool(value: Any, key: str) -> bool:
    """
    Check a BOOLEAN field and return it unchanged.

    Raises:
        ValueError: If the value is not a BOOLEAN
    """
    if not isinstance(value, bool):
        raise _wrong_type(key, "a BOOLEAN", value)
    return value


def validate_str(
    value: Any,
    key: str,
    choices: list[str] | tuple[str, ...] | None = None,
    allow_empty: bool = True,
) -> str:
    """
    Check a string field and return it unchanged.

    Raises:
        ValueError: If the value is not a STRING, is empty when that is not
            allowed, or is outside the allowed choices
    """
    if not isinstance(value, str):
        raise _wrong_type(key, "a STRING", value)
    if not allow_empty and not value.strip():
        raise ValueError(f"'{key}' cannot be empty")
    if choices is not None and value not in choices:
        raise ValueError(
            f"'{key}' must be one of: {_listed(choices)}. Got {value!r}"
        )
    return value


def validate_color_code(value: Any, key: str) -> str | None:
    """
    Check a hex colour and return it normalised, or None for a random colour.

    Empty means 'pick one at generation time', which is how ImageConfig reads
    an unset colour, so blank input is stored as None rather than "".

    Raises:
        ValueError: If the value is not a hex colour
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise _wrong_type(key, "a hex colour such as '#FF0000'", value)

    digits = value.strip().lstrip("#")
    if not digits:
        return None
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    if len(digits) != 6:
        raise ValueError(
            f"'{key}' must be a hex colour such as '#FF0000', got {value!r}"
        )
    try:
        int(digits, 16)
    except ValueError:
        raise ValueError(
            f"'{key}' must be a hex colour such as '#FF0000', got {value!r}"
        )
    return f"#{digits.upper()}"


def validate_image(value: Any, key: str) -> torch.Tensor:
    """
    Check an IMAGE field and return it unchanged.

    Raises:
        ValueError: If the value is not an IMAGE tensor
    """
    if not isinstance(value, torch.Tensor):
        raise _wrong_type(key, "an IMAGE tensor", value)
    return value


def validate_latent(value: Any, key: str) -> dict[str, Any]:
    """
    Check a LATENT field and return it unchanged.

    Raises:
        ValueError: If the value is not a LATENT dict with tensor samples
    """
    if not isinstance(value, dict) or "samples" not in value:
        raise _wrong_type(key, "a LATENT dict holding 'samples'", value)
    if not isinstance(value["samples"], torch.Tensor):
        raise _wrong_type(key, "a LATENT dict whose 'samples' is a tensor", value)
    return value


def validate_lora_list(value: Any, key: str) -> list[LoraSpec]:
    """
    Check a whole LoRA list field and return it unchanged.

    Raises:
        ValueError: If the value is not a list of LoraSpec
    """
    if not isinstance(value, list):
        raise _wrong_type(key, "a list of LoraSpec", value)
    for spec in value:
        if not isinstance(spec, LoraSpec):
            raise _wrong_type(key, "a list of LoraSpec", value)
    return value
