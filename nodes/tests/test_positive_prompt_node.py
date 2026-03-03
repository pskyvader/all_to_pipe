"""
Unit tests for PositivePromptNode.

Tests the positive prompt assignment node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..positive_prompt_node import PositivePromptNode
from ...alltopipe_types import Pipe, PositivePrompt


class TestPositivePromptNode(unittest.TestCase):
    """Test suite for PositivePromptNode."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipe = Pipe()

    def test_node_has_required_inputs(self):
        """Test that PositivePromptNode has optional pipe and required feature/text."""
        node = PositivePromptNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        self.assertIn("pipe", optional)
        
        required = inputs.get("required", {})
        self.assertIn("feature", required)
        self.assertIn("text", required)

    def test_node_class_name(self):
        """Test PositivePromptNode class attributes."""
        self.assertEqual(PositivePromptNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(PositivePromptNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(PositivePromptNode.FUNCTION, "execute")
        self.assertEqual(PositivePromptNode.CATEGORY, "all-to-pipe")


if __name__ == "__main__":
    unittest.main()
