"""
All-to-Pipe update pipe node.

Writes a single element into a Pipe, leaving everything else as it was.
"""

from typing import Any
from ..alltopipe_types import Pipe
from ..common.elements import ElementRegistry


class UpdatePipeNode:
    """
    Writes a single element into a Pipe.

    The key is the dotted path of the field, so every field the Pipe holds can
    be overridden on its own: "parameters.steps", "image_config.width",
    "positive_prompt.age", "loras[0].weight" and so on.

    The Pipe that comes in is never touched: the write happens on a copy, and
    anything the write invalidates (a parsed prompt, a cached model, an
    encoded latent) is rebuilt or dropped so the Pipe stays self-consistent.
    """

    def __init__(self) -> None:
        """Initialize the pipe node."""
        pass

    @staticmethod
    def execute(key: str, value: Any, pipe: Pipe | None = None) -> tuple[Pipe]:
        """
        Execute the node and set one element of the pipe.

        Args:
            key: Dotted path of the element, e.g. "parameters.steps"
            value: Value to store in the field
            pipe: Optional Pipe instance (creates new if None)

        Returns:
            Tuple containing the modified Pipe instance

        Raises:
            ValueError: If the key is not a writable element, or it cannot be built
        """
        target: Pipe = pipe if pipe is not None else Pipe()

        return (ElementRegistry.write(target, key, value),)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "key": (
                    ElementRegistry.update_keys(),
                    {
                        "tooltip": "Dotted path of the Pipe field to write, for "
                        "example parameters.steps. List entries carry their "
                        "index, for example loras[0].weight. A field derived "
                        "from others, such as a parsed template, cannot be set.",
                    },
                ),
                "value": "*",
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"