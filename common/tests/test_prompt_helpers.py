"""
Unit tests for prompt_helpers module.

Tests prompt parsing and merging functions.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.prompt_helpers import prompt_to_string, merge_prompts
from alltopipe_types import PositivePrompt, NegativePrompt


class TestPromptHelpers(unittest.TestCase):
    """Test suite for prompt helper functions."""

    def test_prompt_to_string_positive(self):
        """Test converting PositivePrompt to dictionary."""
        prompt = PositivePrompt()
        prompt.age = "young"
        prompt.body = "athletic"

        result = prompt_to_string(prompt)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("age"), "young")
        self.assertEqual(result.get("body"), "athletic")

    def test_prompt_to_string_negative(self):
        """Test converting NegativePrompt to dictionary."""
        prompt = NegativePrompt()
        prompt.static = "blurry"
        prompt.lora = "bad_lora"

        result = prompt_to_string(prompt)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("static"), "blurry")
        self.assertEqual(result.get("lora"), "bad_lora")

    def test_merge_prompts_positive(self):
        """Test merging positive prompts."""
        prompt1 = {"age": "young", "body": "athletic"}
        prompt2 = {"clothes": "casual", "background": "outdoor"}

        result = merge_prompts(prompt1, prompt2)
        self.assertIsInstance(result, str)
        # Result should contain values from both prompts
        self.assertIn("young", result)
        self.assertIn("casual", result)

    def test_merge_prompts_empty(self):
        """Test merging with empty prompts."""
        prompt1 = {}
        prompt2 = {"age": "old"}

        result = merge_prompts(prompt1, prompt2)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
