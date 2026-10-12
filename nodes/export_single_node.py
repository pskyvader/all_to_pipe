"""
All-to-Pipe export single node.

Resolves one element of a Pipe into a value other nodes can consume.
"""

from typing import Any
from ..alltopipe_types import Pipe
from ..common.elements import ElementRegistry


class ExportSingleNode:
    """
    Reads a single element out of a Pipe.

    The key is the dotted path of the field, so every field the Pipe holds can
    be wired on its own: "parameters.steps", "image_config.width",
    "positive_prompt.age", "loras[0].weight" and so on.
    """

    def __init__(self) -> None:
        """Initialize the export node."""
        pass

    @staticmethod
    def execute(pipe: Pipe, key: str) -> tuple[Any, ...]:
        """
        Execute the node and export one element of the pipe.

        Args:
            pipe: The Pipe to read from
            key: Dotted path of the element, e.g. "parameters.steps"

        Returns:
            Tuple holding the single exported value

        Raises:
            ValueError: If the key is not a readable element, or it is not set
        """
        return (ElementRegistry.read(pipe, key),)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": (
                    ElementRegistry.export_keys(),
                    {
                        "tooltip": "Dotted path of the Pipe field to read, for example "
                        "parameters.steps. List entries carry their index, for "
                        "example loras[0].weight. Reading a field that the Pipe "
                        "has not got raises an error.",
                    },
                ),
            }
        }

    RETURN_TYPES: tuple[Any] = ("*",)
    RETURN_NAMES: tuple[str] = ("value",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"