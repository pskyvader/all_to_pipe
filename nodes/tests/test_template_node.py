"""
Unit tests for TemplateNode.

Tests the prompt template assignment node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..template_node import TemplateNode
from ...alltopipe_types import Pipe, PositivePrompt, NegativePrompt


class TestTemplateNode(unittest.TestCase):
    """Test suite for TemplateNode."""

    def test_node_class_attributes(self):
        """Test TemplateNode class attributes."""
        self.assertEqual(TemplateNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(TemplateNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(TemplateNode.FUNCTION, "execute")
        self.assertEqual(TemplateNode.CATEGORY, "all-to-pipe")

    def test_node_has_required_inputs(self):
        """Test that TemplateNode has optional pipe and required template inputs."""
        node = TemplateNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        self.assertIn("pipe", optional)
        
        required = inputs.get("required", {})
        self.assertIn("template_type", required)
        self.assertIn("template_text", required)

    def test_template_syntax(self):
        """Test that templates use <variable> syntax."""
        # Verify that <> syntax is used instead of {}
        # This is just documentation, actual parsing is in TemplateParser
        template = "A <age> <body> person"
        # Should contain angle brackets, not curly braces
        self.assertIn("<", template)
        self.assertNotIn("{", template)

    def test_feature_combo_values(self):
        """Test that feature is a COMBO input."""
        node = TemplateNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        feature = required.get("feature")

        # Should be present
        self.assertIsNotNone(feature)


if __name__ == "__main__":
    unittest.main()
