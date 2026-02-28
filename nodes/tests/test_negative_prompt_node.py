"""
Unit tests for NegativePromptNode.

Tests the negative prompt assignment node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.negative_prompt_node import NegativePromptNode
from alltopipe_types import Pipe, NegativePrompt


class TestNegativePromptNode(unittest.TestCase):
    """Test suite for NegativePromptNode."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipe = Pipe()

    def test_node_has_required_inputs(self):
        """Test that NegativePromptNode has required inputs."""
        node = NegativePromptNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)
        self.assertIn("static", required)

    def test_node_class_name(self):
        """Test NegativePromptNode class attributes."""
        self.assertEqual(NegativePromptNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(NegativePromptNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(NegativePromptNode.FUNCTION, "execute")
        self.assertEqual(NegativePromptNode.CATEGORY, "All-to-Pipe")

    def test_node_optional_inputs(self):
        """Test that NegativePromptNode has optional inputs."""
        node = NegativePromptNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        # lora and model should be optional
        if optional:
            self.assertIn("lora", optional)
            self.assertIn("model", optional)


if __name__ == "__main__":
    unittest.main()
