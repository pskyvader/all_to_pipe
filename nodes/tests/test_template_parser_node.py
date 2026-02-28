"""
Unit tests for TemplateParserNode.

Tests the template parsing node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.template_parser_node import TemplateParserNode
from alltopipe_types import Pipe, PositivePrompt


class TestTemplateParserNode(unittest.TestCase):
    """Test suite for TemplateParserNode."""

    def test_node_class_attributes(self):
        """Test TemplateParserNode class attributes."""
        self.assertEqual(TemplateParserNode.RETURN_TYPES, ("STRING",))
        self.assertEqual(TemplateParserNode.RETURN_NAMES, ("parsed_template",))
        self.assertEqual(TemplateParserNode.FUNCTION, "execute")
        self.assertEqual(TemplateParserNode.CATEGORY, "All-to-Pipe")

    def test_node_has_required_inputs(self):
        """Test that TemplateParserNode has required inputs."""
        node = TemplateParserNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)
        self.assertIn("template", required)

    def test_node_optional_inputs(self):
        """Test optional inputs for TemplateParserNode."""
        node = TemplateParserNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        if optional:
            self.assertIn("allow_missing", optional)
            self.assertIn("default_value", optional)


if __name__ == "__main__":
    unittest.main()
