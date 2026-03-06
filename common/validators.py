"""
All-to-Pipe validators module.

Cross-node validation functions.
"""

from ..alltopipe_types import Pipe
from ..alltopipe_types import Model
from ..alltopipe_types import Parameters
from ..alltopipe_types import ImageConfig



def validate_pipe(pipe: Pipe) -> None:
    """
    Validate that a Pipe contains all required fields.
    
    This is a cross-node validator used by export nodes.
    
    Raises:
        ValueError: If required fields are missing or invalid
        
    Args:
        pipe: The Pipe instance to validate
    """
    if not isinstance(pipe.model,Model):
        raise ValueError("Pipe.model must be a Model instance")
    if not isinstance(pipe.parameters,Parameters):
        raise ValueError("Pipe.parameters must be a Parameters instance")
    if not isinstance(pipe.image_config,ImageConfig):
        raise ValueError("Pipe.image_config must be an ImageConfig instance")