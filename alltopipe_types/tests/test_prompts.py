"""
Unit tests for Prompt types.

Tests the PositivePrompt and NegativePrompt classes and PromptProcessor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import PositivePrompt, NegativePrompt


class TestPositivePrompt(unittest.TestCase):
    """Test suite for PositivePrompt type."""

    def test_positive_prompt_creation(self):
        """Test creating a PositivePrompt instance."""
        prompt = PositivePrompt()
        self.assertIsNone(prompt.age)
        self.assertIsNone(prompt.body)
        self.assertIsNone(prompt.clothes)

    def test_positive_prompt_attributes(self):
        """Test PositivePrompt allowed attributes."""
        prompt = PositivePrompt()
        prompt.age = "young"
        prompt.body = "slim"
        prompt.clothes = "casual"
        prompt.background = "outdoor"
        prompt.face = "smiling"
        prompt.lora = "lora_style"
        prompt.model = "model_style"

        self.assertEqual(prompt.age, "young")
        self.assertEqual(prompt.body, "slim")
        self.assertEqual(prompt.clothes, "casual")
        self.assertEqual(prompt.background, "outdoor")
        self.assertEqual(prompt.face, "smiling")
        self.assertEqual(prompt.lora, "lora_style")
        self.assertEqual(prompt.model, "model_style")

    def test_positive_prompt_partial_attributes(self):
        """Test PositivePrompt with only some attributes set."""
        prompt = PositivePrompt()
        prompt.age = "old"
        prompt.face = "wrinkled"

        self.assertEqual(prompt.age, "old")
        self.assertEqual(prompt.face, "wrinkled")
        self.assertIsNone(prompt.body)
        self.assertIsNone(prompt.clothes)

    def test_positive_prompt_empty_string_attributes(self):
        """Test PositivePrompt with empty string attributes."""
        prompt = PositivePrompt()
        prompt.age = ""
        prompt.body = ""

        self.assertEqual(prompt.age, "")
        self.assertEqual(prompt.body, "")


class TestNegativePrompt(unittest.TestCase):
    """Test suite for NegativePrompt type."""

    def test_negative_prompt_creation(self):
        """Test creating a NegativePrompt instance."""
        prompt = NegativePrompt()
        self.assertIsNone(prompt.static)
        self.assertIsNone(prompt.lora)
        self.assertIsNone(prompt.model)

    def test_negative_prompt_attributes(self):
        """Test NegativePrompt allowed attributes."""
        prompt = NegativePrompt()
        prompt.static = "blurry"
        prompt.lora = "bad_lora"
        prompt.model = "bad_model"

        self.assertEqual(prompt.static, "blurry")
        self.assertEqual(prompt.lora, "bad_lora")
        self.assertEqual(prompt.model, "bad_model")

    def test_negative_prompt_multiple_values(self):
        """Test NegativePrompt with multiple negative descriptions."""
        prompt = NegativePrompt()
        prompt.static = "blurry, pixelated, low quality"
        prompt.lora = "bad_style"
        prompt.model = "wrong_model"

        self.assertIn("blurry", prompt.static)
        self.assertIn("pixelated", prompt.static)

    def test_negative_prompt_partial_attributes(self):
        """Test NegativePrompt with only some attributes set."""
        prompt = NegativePrompt()
        prompt.static = "blurry"

        self.assertEqual(prompt.static, "blurry")
        self.assertIsNone(prompt.lora)
        self.assertIsNone(prompt.model)


if __name__ == "__main__":
    unittest.main()
