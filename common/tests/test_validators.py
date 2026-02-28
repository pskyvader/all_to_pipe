"""
Unit tests for validators module.

Tests validation functions.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.validators import validate_pipe
from alltopipe_types import Pipe, Model, Parameters


class TestValidators(unittest.TestCase):
    """Test suite for validators."""

    def test_validate_pipe_complete(self):
        """Test validating a complete pipe."""
        pipe = Pipe(
            model=Model(name="test.safetensors", subfolder="checkpoints"),
            parameters=Parameters(
                steps=20,
                cfg=7.5,
                sampler="euler",
                scheduler="karras",
                seed=12345,
            ),
        )

        is_valid, errors = validate_pipe(pipe)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_pipe_missing_model(self):
        """Test validation fails without model."""
        pipe = Pipe()
        pipe.parameters = Parameters(
            steps=20,
            cfg=7.5,
            sampler="euler",
            scheduler="karras",
            seed=12345,
        )

        is_valid, errors = validate_pipe(pipe)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_validate_pipe_missing_parameters(self):
        """Test validation fails without parameters."""
        pipe = Pipe(
            model=Model(name="test.safetensors", subfolder="checkpoints")
        )

        is_valid, errors = validate_pipe(pipe)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_validate_pipe_missing_both(self):
        """Test validation fails without model and parameters."""
        pipe = Pipe()

        is_valid, errors = validate_pipe(pipe)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
