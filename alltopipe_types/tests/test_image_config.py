"""
Unit tests for ImageConfig type.

Tests the ImageConfig class and ImageConfigProcessor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import ImageConfig


class TestImageConfig(unittest.TestCase):
    """Test suite for ImageConfig type."""

    def test_image_config_creation(self):
        """Test creating an ImageConfig instance."""
        config = ImageConfig(
            width=512,
            height=768,
            batch_size=2,
            noise=0.8,
            color_code="#FF0000",
        )
        self.assertEqual(config.width, 512)
        self.assertEqual(config.height, 768)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.noise, 0.8)
        self.assertEqual(config.color_code, "#FF0000")

    def test_image_config_noise_clamping(self):
        """Test that noise is clamped to 0-1 range."""
        # Test noise > 1.0 is clamped
        config = ImageConfig(width=512, height=512, batch_size=1, noise=2.0)
        self.assertEqual(config.noise, 1.0)

        # Test noise < 0 is clamped
        config = ImageConfig(width=512, height=512, batch_size=1, noise=-0.5)
        self.assertEqual(config.noise, 0.0)

        # Test valid noise
        config = ImageConfig(width=512, height=512, batch_size=1, noise=0.5)
        self.assertEqual(config.noise, 0.5)

    def test_image_config_optional_color(self):
        """Test ImageConfig with optional color code."""
        config = ImageConfig(width=512, height=512, batch_size=1, color_code=None)
        self.assertIsNone(config.color_code)

    def test_image_config_standard_sizes(self):
        """Test ImageConfig with standard image sizes."""
        # 1024x1024
        config = ImageConfig(width=1024, height=1024, batch_size=1)
        self.assertEqual(config.width, 1024)
        self.assertEqual(config.height, 1024)

        # 512x768
        config = ImageConfig(width=512, height=768, batch_size=1)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.height, 768)

        # 768x512
        config = ImageConfig(width=768, height=512, batch_size=1)
        self.assertEqual(config.width, 768)
        self.assertEqual(config.height, 512)

    def test_image_config_batch_sizes(self):
        """Test ImageConfig with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16, 32]:
            config = ImageConfig(width=512, height=512, batch_size=batch_size)
            self.assertEqual(config.batch_size, batch_size)


if __name__ == "__main__":
    unittest.main()
