"""
Unit tests for LoraSpec type.

Tests the LoraSpec class and LoraProcessor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import LoraSpec


class TestLoraSpec(unittest.TestCase):
    """Test suite for LoraSpec type."""

    def test_lora_creation(self):
        """Test creating a LoraSpec instance."""
        lora = LoraSpec(
            name="lora.safetensors",
            subfolder="loras",
            weight=0.8,
            clip_weight=0.5,
        )
        self.assertEqual(lora.name, "lora.safetensors")
        self.assertEqual(lora.subfolder, "loras")
        self.assertEqual(lora.weight, 0.8)
        self.assertEqual(lora.clip_weight, 0.5)

    def test_lora_weight_range(self):
        """Test that LoRA weights are properly stored."""
        lora = LoraSpec(
            name="test.safetensors",
            subfolder="loras",
            weight=1.5,  # Can be > 1.0
            clip_weight=-0.5,  # Can be negative
        )
        self.assertEqual(lora.weight, 1.5)
        self.assertEqual(lora.clip_weight, -0.5)

    def test_lora_zero_weight(self):
        """Test LoRA with zero weight."""
        lora = LoraSpec(
            name="test.safetensors",
            subfolder="loras",
            weight=0.0,
            clip_weight=0.0,
        )
        self.assertEqual(lora.weight, 0.0)
        self.assertEqual(lora.clip_weight, 0.0)

    def test_lora_max_weight(self):
        """Test LoRA with maximum weight."""
        lora = LoraSpec(
            name="test.safetensors",
            subfolder="loras",
            weight=2.0,
            clip_weight=2.0,
        )
        self.assertEqual(lora.weight, 2.0)
        self.assertEqual(lora.clip_weight, 2.0)

    def test_lora_min_weight(self):
        """Test LoRA with minimum weight."""
        lora = LoraSpec(
            name="test.safetensors",
            subfolder="loras",
            weight=-2.0,
            clip_weight=-2.0,
        )
        self.assertEqual(lora.weight, -2.0)
        self.assertEqual(lora.clip_weight, -2.0)


if __name__ == "__main__":
    unittest.main()
