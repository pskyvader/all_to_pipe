"""
Unit tests for file_helpers module.

Tests file discovery and path resolution.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.file_helpers import discover_model_subfolders, discover_lora_subfolders


class TestFileHelpers(unittest.TestCase):
    """Test suite for file discovery functions."""

    def test_discover_model_subfolders_returns_list(self):
        """Test that discover_model_subfolders returns a list."""
        result = discover_model_subfolders()
        self.assertIsInstance(result, list)

    def test_discover_model_subfolders_contains_strings(self):
        """Test that model subfolders are all strings."""
        result = discover_model_subfolders()
        for subfolder in result:
            self.assertIsInstance(subfolder, str)

    def test_discover_model_subfolders_includes_root(self):
        """Test that model subfolders include root directory."""
        result = discover_model_subfolders()
        self.assertIn("", result)  # Empty string represents root

    def test_discover_lora_subfolders_returns_list(self):
        """Test that discover_lora_subfolders returns a list."""
        result = discover_lora_subfolders()
        self.assertIsInstance(result, list)

    def test_discover_lora_subfolders_contains_strings(self):
        """Test that LoRA subfolders are all strings."""
        result = discover_lora_subfolders()
        for subfolder in result:
            self.assertIsInstance(subfolder, str)

    def test_discover_lora_subfolders_includes_root(self):
        """Test that LoRA subfolders include root directory."""
        result = discover_lora_subfolders()
        self.assertIn("", result)  # Empty string represents root


if __name__ == "__main__":
    unittest.main()
