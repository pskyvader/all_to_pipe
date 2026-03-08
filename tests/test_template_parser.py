"""
Unit tests for All-to-Pipe template system.

Tests TemplateParser functionality including:
- Placeholder detection
- Template validation
- Template parsing with variable substitution
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        template = "A {age} {body} person"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(placeholders, ["age", "body"])

    def test_find_placeholders_complex(self):
        """Test finding placeholders in a complex template."""
        template = "A {age} {body} wearing {clothes} in {background}"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(
            placeholders, ["age", "body", "clothes", "background"]
        )

    def test_find_placeholders_empty(self):
        """Test finding placeholders in a template with none."""
        template = "A simple template with no placeholders"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(placeholders, [])

    def test_find_placeholders_duplicates(self):
        """Test that duplicate placeholders are all found."""
        template = "{age} and {age} again"
        placeholders = TemplateParser.find_placeholders(template)
        self.assertEqual(placeholders, ["age", "age"])


    def test_parse_template_simple(self):
        """Test parsing a simple template."""
        template = "A {age} person"
        result = TemplateParser.parse_template(template, self.positive_prompt)
        self.assertEqual(result, "A young person")

    def test_parse_template_complex(self):
        """Test parsing a complex template."""
        template = "A {age} {body} person wearing {clothes} in {background}"
        result = TemplateParser.parse_template(template, self.positive_prompt)
        expected = "A young athletic person wearing casual shirt in outdoor garden"
        self.assertEqual(result, expected)

    def test_parse_template_missing_variable_error(self):
        """Test that parsing with missing variable raises error by default."""
        template = "A {age} {nonexistent} person"
        with self.assertRaises(ValueError):
            TemplateParser.parse_template(template, self.positive_prompt)

    def test_parse_template_missing_variable_allowed(self):
        """Test parsing with missing variable when allow_missing=True."""
        template = "A {age} {nonexistent} person"
        result = TemplateParser.parse_template(
            template, self.positive_prompt, allow_missing=True
        )
        self.assertEqual(result, "A young [MISSING] person")

    def test_parse_template_custom_default_value(self):
        """Test parsing with custom default value for missing variables."""
        template = "A {age} {nonexistent} person"
        result = TemplateParser.parse_template(
            template, self.positive_prompt, allow_missing=True, default_value="???"
        )
        self.assertEqual(result, "A young ??? person")

    def test_parse_template_empty_string(self):
        """Test parsing an empty template."""
        result = TemplateParser.parse_template("", self.positive_prompt)
        self.assertEqual(result, "")

    def test_parse_template_no_placeholders(self):
        """Test parsing a template with no placeholders."""
        template = "This is a normal sentence"
        result = TemplateParser.parse_template(template, self.positive_prompt)
        self.assertEqual(result, template)

    def test_get_required_variables(self):
        """Test extracting required variables from template."""
        template = "A {age} {body} person wearing {clothes}"
        variables = TemplateParser.get_required_variables(template)
        self.assertEqual(variables, ["age", "body", "clothes"])

    def test_get_required_variables_no_duplicates(self):
        """Test that duplicates are removed."""
        template = "A {age} and {age} again"
        variables = TemplateParser.get_required_variables(template)
        self.assertEqual(variables, ["age"])


    def test_parse_from_negative_prompt(self):
        """Test parsing uses values from negative prompt if not in positive."""
        template = "Avoid {static}"
        result = TemplateParser.parse_template(
            template, self.positive_prompt, self.negative_prompt
        )
        self.assertEqual(result, "Avoid blurry")

    def test_error_invalid_template_type(self):
        """Test that non-string template raises error."""
        with self.assertRaises(ValueError):
            TemplateParser.find_placeholders(123)


    def test_error_invalid_template_for_parse(self):
        """Test that non-string template raises error in parse."""
        with self.assertRaises(ValueError):
            TemplateParser.parse_template(123, self.positive_prompt)


class TestTemplateParserEdgeCases(unittest.TestCase):
    """Test edge cases for TemplateParser."""

    def test_nested_braces(self):
        """Test handling of nested braces (should not be treated as placeholders)."""
        template = "A {age} person {{with braces}}"
        placeholders = TemplateParser.find_placeholders(template)
        # Only {age} should be found, not {with braces}
        self.assertIn("age", placeholders)

    def test_special_characters_in_placeholder(self):
        """Test placeholders with underscores and numbers."""
        positive_prompt = PositivePrompt()
        positive_prompt.feature_1 = "value1"
        positive_prompt.age_group = "young"
        
        template = "A {age_group} person with {feature_1}"
        result = TemplateParser.parse_template(template, positive_prompt)
        self.assertEqual(result, "A young person with value1")

    def test_whitespace_in_template(self):
        """Test templates with various whitespace."""
        positive_prompt = PositivePrompt()
        positive_prompt.age = "young"
        positive_prompt.body = "slim"
        
        template = "A {age}   {body}    person"
        result = TemplateParser.parse_template(template, positive_prompt)
        self.assertEqual(result, "A young   slim    person")


if __name__ == "__main__":
    unittest.main()
