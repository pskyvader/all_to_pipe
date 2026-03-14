"""
All-to-Pipe export node.

Resolves Pipe into sampler-ready objects.
"""

from typing import Dict, Any, Tuple
from ..alltopipe_types import Pipe


class ExportSingleNode:

    elements_to_update: Dict[str, Dict[str, Any]] = {
        "image": {
            "attr_name": "image",  # The attribute inside ImageConfig
            "pipe_key": "image_config",  # The attribute inside Pipe
        }
    }

    def __init__(self) -> None:
        """Initialize the export node."""
        pass

    @staticmethod
    def execute(pipe: Pipe, key: str) -> Tuple[Any]:
        element = ExportSingleNode.elements_to_update[key]

        export_object = getattr(pipe, element["pipe_key"], None)
        if export_object is None:
            raise ValueError(f"Pipe does not have a {element['pipe_key']} attribute")
        export_value = getattr(export_object, element["attr_name"], None)
        if export_value is None:
            raise ValueError(
                f"attribute {element['attr_name']} in {element['pipe_key']}  is None"
            )

        return (export_value,)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": (list(cls.elements_to_update.keys()),),
            }
        }

    RETURN_TYPES: Tuple[Any] = ("*",)
    RETURN_NAMES: Tuple[str] = ("value",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
