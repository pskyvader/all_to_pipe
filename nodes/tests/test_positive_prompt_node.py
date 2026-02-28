"""
Unit tests for PositivePromptNode.

Tests the positive prompt assignment node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.positive_prompt_node import PositivePromptNode
from alltopipe_types import Pipe, PositivePrompt


class TestPositivePromptNode(unittest.TestCase):
    """Test suite for PositivePromptNode."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipe = Pipe()

    def test_node_has_required_inputs(self):
        """Test that PositivePromptNode has required inputs."""
        node = PositivePromptNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)
        self.assertIn("age", required)
        self.assertIn("body", required)
        self.assertIn("clothes", required)

    def test_node_class_name(self):
        """Test PositivePromptNode class attributes."""
        self.assertEqual(PositivePromptNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(PositivePromptNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(PositivePromptNode.FUNCTION, "execute")
        self.assertEqual(PositivePromptNode.CATEGORY, "All-to-Pipe")


if __name__ == "__main__":
    unittest.main()
