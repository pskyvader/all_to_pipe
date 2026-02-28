"""
All-to-Pipe pipe node.

Creates an empty Pipe object as the entry point of the pipeline.
"""

from typing import Dict, Any, Tuple, Optional
from ..alltopipe_types import Pipe

#TODO: SINCE all nodes should create a pipe when no pipe is passed, this node is unnecessary 

class PipeNode:
    """
    Creates an empty Pipe object.
    
    This is the entry point of the pipeline.
    Produces a new Pipe instance with no defaults set.
    Can be used standalone or to reset a pipe mid-pipeline.
    """

    def __init__(self) -> None:
        """Initialize the pipe node."""
        pass

    @staticmethod
    def execute(pipe: Optional[Pipe] = None) -> Tuple[Pipe]:
        """
        Execute the node and create a new empty Pipe.
        
        Args:
            pipe: Optional existing pipe (ignored, always creates new)

        Returns:
            Tuple containing a single new Pipe instance
        """
        new_pipe: Pipe = Pipe()
        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.
        
        Returns:
            Dictionary defining node inputs (pipe is optional)
        """
        return {
            "optional": {
                "pipe": ("PIPE",),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
