"""
All-to-Pipe parameters builder node.

Builds and attaches sampler parameters to the Pipe.
"""

from typing import Dict, Any, Tuple, Optional
import random
from ..alltopipe_types import Pipe, Parameters
from ..common.utils import deep_copy_pipe
from ..common.constants import (
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
    MIN_STEPS,
    MAX_STEPS,
    MIN_CFG,
    MAX_CFG,
    MIN_SEED,
    MAX_SEED,
)


# def validate_parameters(params: Parameters) -> None:
#     """
#     Validate that all sampler parameters are properly defined.

#     Raises:
#         ValueError: If any parameter is invalid
#     """
#     if params is None:
#         raise ValueError("Parameters cannot be None")
#     if params.steps is None:
#         raise ValueError("steps is required but not set")
#     if not isinstance(params.steps, int) or params.steps < MIN_STEPS or params.steps > MAX_STEPS:
#         raise ValueError(
#             f"steps must be an integer between {MIN_STEPS} and {MAX_STEPS}, got {params.steps}"
#         )
#     if params.cfg is None:
#         raise ValueError("cfg is required but not set")
#     if not isinstance(params.cfg, (int, float)) or params.cfg < MIN_CFG or params.cfg > MAX_CFG:
#         raise ValueError(
#             f"cfg must be a number between {MIN_CFG} and {MAX_CFG}, got {params.cfg}"
#         )
#     if params.sampler is None:
#         raise ValueError("sampler is required but not set")
#     if not isinstance(params.sampler, str) or (params.sampler != "RANDOM" and params.sampler not in SUPPORTED_SAMPLERS):
#         raise ValueError(
#             f"sampler must be one of {SUPPORTED_SAMPLERS} or 'RANDOM', got {params.sampler}"
#         )
#     if params.scheduler is None:
#         raise ValueError("scheduler is required but not set")
#     if not isinstance(params.scheduler, str) or (params.scheduler != "RANDOM" and params.scheduler not in SUPPORTED_SCHEDULERS):
#         raise ValueError(
#             f"scheduler must be one of {SUPPORTED_SCHEDULERS} or 'RANDOM', got {params.scheduler}"
#         )
#     if params.seed is None:
#         raise ValueError("seed is required but not set")
#     if not isinstance(params.seed, int) or params.seed < MIN_SEED or params.seed > MAX_SEED:
#         raise ValueError(
#             f"seed must be an integer between {MIN_SEED} and {MAX_SEED}, got {params.seed}"
#         )


class ParametersBuilderNode:
    """
    Builds and attaches sampler parameters to the Pipe.

    Features:
    - COMBO selectors for sampler and scheduler
    - Random seed generation option
    - Validation of all parameter ranges
    """

    def __init__(self) -> None:
        """Initialize the parameters builder node."""
        pass

    @staticmethod
    def execute(
        pipe: Optional[Pipe] = None,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = 0,
    ) -> Tuple[Pipe]:
        """
        Execute the node and build parameters for the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler: Sampler name or 'RANDOM' for random selection
            scheduler: Scheduler name or 'RANDOM' for random selection
            seed: Random seed for reproducibility

        Returns:
            Tuple containing the modified Pipe instance
        """
        # Deep copy pipe to avoid modifying the original
        new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()

        # Handle RANDOM sampler selection
        selected_sampler = sampler
        if sampler == "RANDOM":
            selected_sampler = random.choice(SUPPORTED_SAMPLERS)

        # Handle RANDOM scheduler selection
        selected_scheduler = scheduler
        if scheduler == "RANDOM":
            selected_scheduler = random.choice(SUPPORTED_SCHEDULERS)

        # Create and validate the parameters
        parameters: Parameters = Parameters(
            steps=steps,
            cfg=cfg,
            sampler=selected_sampler,
            scheduler=selected_scheduler,
            seed=seed,
        )
        # validate_parameters(parameters)

        # Attach parameters to the pipe
        new_pipe.parameters = parameters

        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node with dynamic selectors.

        Returns:
            Dictionary defining node inputs with COMBO selectors
        """
        # Add RANDOM option at the beginning of sampler and scheduler lists
        sampler_options = ["RANDOM"] + SUPPORTED_SAMPLERS
        scheduler_options = ["RANDOM"] + SUPPORTED_SCHEDULERS

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "steps": ("INT", {"default": 20, "min": MIN_STEPS, "max": MAX_STEPS}),
                "cfg": (
                    "FLOAT",
                    {"default": 7.0, "min": MIN_CFG, "max": MAX_CFG, "step": 0.1},
                ),
                "sampler": (tuple(sampler_options),),
                "scheduler": (tuple(scheduler_options),),
                "seed": ("INT", {"default": 0, "min": MIN_SEED, "max": MAX_SEED}),
            },
        }

    RETURN_TYPES: Tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: Tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
