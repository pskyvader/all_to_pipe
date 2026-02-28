"""
Unit tests for LoraNode.

Tests the LoRA selection and application node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.lora_node import LoraNode
from alltopipe_types import Pipe


class TestLoraNode(unittest.TestCase):
    """Test suite for LoraNode."""

    def test_node_class_attributes(self):
        """Test LoraNode class attributes."""
        self.assertEqual(LoraNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(LoraNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(LoraNode.FUNCTION, "execute")
        self.assertEqual(LoraNode.CATEGORY, "All-to-Pipe")

    def test_node_has_required_inputs(self):
        """Test that LoraNode has required inputs."""
        node = LoraNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)
        self.assertIn("lora_subfolder", required)
        self.assertIn("lora_name", required)
        self.assertIn("weight", required)
        self.assertIn("clip_weight", required)
        self.assertIn("random_lora", required)

    def test_node_weight_inputs(self):
        """Test weight input specifications."""
        node = LoraNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        weight = required.get("weight", {})
        clip_weight = required.get("clip_weight", {})

        # Both should be present
        self.assertIsNotNone(weight)
        self.assertIsNotNone(clip_weight)


if __name__ == "__main__":
    unittest.main()
