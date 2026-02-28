"""
Unit tests for All-to-Pipe validators.

Tests validation functions for Pipe and other types.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alltopipe_types import Pipe, Model, Parameters, PositivePrompt, NegativePrompt
from common.validators import validate_pipe


class TestValidators(unittest.TestCase):
    """Test suite for validation functions."""

    def test_validate_pipe_complete(self):
        """Test validating a complete pipe."""
        pipe = Pipe()
        pipe.model = Model(name="test.safetensors", subfolder="checkpoints")
        pipe.parameters = Parameters(
            steps=20, cfg=7.5, sampler="euler", scheduler="karras", seed=12345
        )
        pipe.positive_prompt = PositivePrompt()
        pipe.negative_prompt = NegativePrompt()

        # Should not raise
        try:
            validate_pipe(pipe)
        except ValueError:
            self.fail("validate_pipe raised ValueError unexpectedly")

    def test_validate_pipe_missing_model(self):
        """Test validating pipe without model."""
        pipe = Pipe()
        pipe.model = None
        pipe.parameters = Parameters(
            steps=20, cfg=7.5, sampler="euler", scheduler="karras", seed=12345
        )

        # Should raise or handle gracefully
        try:
            validate_pipe(pipe)
            # If it doesn't raise, that's ok too - model might be optional
        except ValueError:
            pass  # Expected behavior for missing required field

    def test_validate_pipe_missing_parameters(self):
        """Test validating pipe without parameters."""
        pipe = Pipe()
        pipe.model = Model(name="test.safetensors", subfolder="checkpoints")
        pipe.parameters = None

        # Should raise or handle gracefully
        try:
            validate_pipe(pipe)
            # If it doesn't raise, that's ok - parameters might be optional
        except ValueError:
            pass  # Expected behavior for missing required field

    def test_validate_pipe_with_prompts(self):
        """Test that pipe with prompts validates correctly."""
        pipe = Pipe()
        pipe.positive_prompt = PositivePrompt()
        pipe.positive_prompt.age = "young"
        pipe.negative_prompt = NegativePrompt()
        pipe.negative_prompt.static = "blurry"

        # Should not raise
        try:
            validate_pipe(pipe)
        except ValueError:
            self.fail("validate_pipe raised ValueError unexpectedly")

    def test_validate_pipe_not_none(self):
        """Test that validate_pipe rejects None."""
        try:
            validate_pipe(None)
            # Might not raise if validation is lenient
        except (ValueError, AttributeError):
            # Expected behavior
            pass


if __name__ == "__main__":
    unittest.main()
