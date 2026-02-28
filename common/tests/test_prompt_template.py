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

    def test_validate_template_all_valid(self):
        """Test validating a template where all variables exist."""
        template = "A <age> <body> person"
        is_valid, missing = TemplateParser.validate_template(
            template, self.positive_prompt, self.negative_prompt
        )
        self.assertTrue(is_valid)
        self.assertEqual(missing, [])

    def test_validate_template_missing_variable(self):
        """Test validating a template with missing variable."""
        template = "A <age> <nonexistent> person"
        is_valid, missing = TemplateParser.validate_template(
            template, self.positive_prompt, self.negative_prompt
        )
        self.assertFalse(is_valid)
        self.assertIn("nonexistent", missing)

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

    def test_parse_multiple_templates(self):
        """Test parsing multiple templates at once."""
        templates = [
            "A <age> person",
            "Wearing <clothes>",
            "In <background>",
        ]
        results = TemplateParser.parse_multiple_templates(
            templates, self.positive_prompt
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "A young person")
        self.assertEqual(results[1], "Wearing casual shirt")
        self.assertEqual(results[2], "In outdoor garden")


if __name__ == "__main__":
    unittest.main()
