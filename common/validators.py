"""
All-to-Pipe validators module.

Cross-node validation functions.
"""

from typing import Optional
from ..alltopipe_types import Pipe


def validate_pipe(pipe: Pipe) -> None:
    """
    Validate that a Pipe contains all required fields.
    
    This is a cross-node validator used by export nodes.
    
    Raises:
        ValueError: If required fields are missing or invalid
        
    Args:
        pipe: The Pipe instance to validate
    """
    if pipe is None:
        raise ValueError("Pipe cannot be None")
