"""
Unit tests for prompt_template module.

Tests the TemplateParser class and template functionality.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.prompt_template import TemplateParser
from alltopipe_types import PositivePrompt, NegativePrompt


class TestTemplateParser(unittest.TestCase):
    """Test suite for TemplateParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.positive_prompt = PositivePrompt()
        self.positive_prompt.age = "young"
        self.positive_prompt.body = "athletic"
        self.positive_prompt.clothes = "casual shirt"
        self.positive_prompt.background = "outdoor garden"
        self.positive_prompt.face = "smiling"

        self.negative_prompt = NegativePrompt()
        self.negative_prompt.static = "blurry"

    def test_find_placeholders_simple(self):
        """Test finding placeholders in a simple template."""
        template = "A <age> <body> person"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(placeholders, ["age", "body"])

    def test_find_placeholders_complex(self):
        """Test finding placeholders in a complex template."""
        template = "A <age> <body> wearing <clothes> in <background>"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(
            placeholders, ["age", "body", "clothes", "background"]
        )

    def test_find_placeholders_empty(self):
        """Test finding placeholders in a template with none."""
        template = "A simple template with no placeholders"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(placeholders, [])

    def test_parse_template_simple(self):
        """Test parsing a simple template."""
        template = "A <age> person"
        result = TemplateParser.parse_template(template, self.positive_prompt)
        self.assertEqual(result, "A young person")

    def test_parse_template_complex(self):
        """Test parsing a complex template."""
        template = "A <age> <body> person wearing <clothes> in <background>"
        result = TemplateParser.parse_template(template, self.positive_prompt)
        expected = "A young athletic person wearing casual shirt in outdoor garden"
        self.assertEqual(result, expected)



if __name__ == "__main__":
    unittest.main()
