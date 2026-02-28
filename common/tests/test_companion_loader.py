"""
Unit tests for companion_loader module.

Tests companion file loading for models and LoRAs.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.companion_loader import load_model_companion, load_lora_companion


class TestCompanionLoader(unittest.TestCase):
    """Test suite for companion file loading."""

    def test_load_model_companion_nonexistent(self):
        """Test loading companion for nonexistent model."""
        # Should return None or empty dict gracefully
        result = load_model_companion("nonexistent_model.safetensors", "")
        self.assertIsNone(result)

    def test_load_lora_companion_nonexistent(self):
        """Test loading companion for nonexistent LoRA."""
        # Should return None or empty dict gracefully
        result = load_lora_companion("nonexistent_lora.safetensors", "")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
