"""
Unit tests for ImageConfigNode.

Tests the image configuration node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.image_config_node import ImageConfigNode
from alltopipe_types import Pipe, ImageConfig


class TestImageConfigNode(unittest.TestCase):
    """Test suite for ImageConfigNode."""

    def test_node_class_attributes(self):
        """Test ImageConfigNode class attributes."""
        self.assertEqual(ImageConfigNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(ImageConfigNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(ImageConfigNode.FUNCTION, "execute")
        self.assertEqual(ImageConfigNode.CATEGORY, "All-to-Pipe")

    def test_node_has_required_inputs(self):
        """Test that ImageConfigNode has required inputs."""
        node = ImageConfigNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)
        self.assertIn("width", required)
        self.assertIn("height", required)
        self.assertIn("batch_size", required)
        self.assertIn("noise", required)

    def test_node_optional_color_code(self):
        """Test that color_code is optional input."""
        node = ImageConfigNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        if optional:
            self.assertIn("color_code", optional)

    def test_node_dimension_inputs(self):
        """Test dimension input specifications."""
        node = ImageConfigNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        width = required.get("width")
        height = required.get("height")

        # Should be present
        self.assertIsNotNone(width)
        self.assertIsNotNone(height)


if __name__ == "__main__":
    unittest.main()
