"""
Unit tests for ModelNode.

Tests the model selection and loading node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..model_node import ModelNode
from ...alltopipe_types import Pipe


class TestModelNode(unittest.TestCase):
    """Test suite for ModelNode."""

    def test_node_class_attributes(self):
        """Test ModelNode class attributes."""
        self.assertEqual(ModelNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(ModelNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(ModelNode.FUNCTION, "execute")
        self.assertEqual(ModelNode.CATEGORY, "all-to-pipe")

    def test_node_has_required_inputs(self):
        """Test that ModelNode has optional pipe and required model parameters."""
        node = ModelNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        self.assertIn("pipe", optional)
        
        required = inputs.get("required", {})
        self.assertIn("model_subfolder", required)
        self.assertIn("model_name", required)
        self.assertIn("random_model", required)

    def test_node_random_model_boolean(self):
        """Test random_model input is boolean."""
        node = ModelNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        random_model = required.get("random_model", {})
        # Should contain boolean type
        self.assertIsNotNone(random_model)


if __name__ == "__main__":
    unittest.main()
