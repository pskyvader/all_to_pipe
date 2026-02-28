"""
Unit tests for Model type.

Tests the Model class and ModelProcessor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import Model


class TestModel(unittest.TestCase):
    """Test suite for Model type."""

    def test_model_creation(self):
        """Test creating a Model instance."""
        model = Model(name="model.safetensors", subfolder="checkpoints")
        self.assertEqual(model.name, "model.safetensors")
        self.assertEqual(model.subfolder, "checkpoints")

    def test_model_attributes(self):
        """Test that Model has expected attributes."""
        model = Model(name="test_model", subfolder="test_folder")
        self.assertTrue(hasattr(model, "name"))
        self.assertTrue(hasattr(model, "subfolder"))

    def test_model_empty_subfolder(self):
        """Test Model with empty subfolder (root)."""
        model = Model(name="model.safetensors", subfolder="")
        self.assertEqual(model.subfolder, "")

    def test_model_nested_subfolder(self):
        """Test Model with nested subfolder."""
        model = Model(name="model.safetensors", subfolder="checkpoints/custom/subfolder")
        self.assertEqual(model.subfolder, "checkpoints/custom/subfolder")


if __name__ == "__main__":
    unittest.main()
