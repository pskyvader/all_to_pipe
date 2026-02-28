"""
Unit tests for ExportNode.

Tests the sampler export node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.export_node import ExportNode
from alltopipe_types import Pipe, Model, Parameters


class TestExportNode(unittest.TestCase):
    """Test suite for ExportNode."""

    def test_node_class_attributes(self):
        """Test ExportNode class attributes."""
        self.assertEqual(ExportNode.RETURN_TYPES, ("MODEL", "CONDITIONING", "CONDITIONING", "STRING"))
        self.assertEqual(ExportNode.FUNCTION, "execute")
        self.assertEqual(ExportNode.CATEGORY, "All-to-Pipe")

    def test_node_has_required_inputs(self):
        """Test that ExportNode has required inputs."""
        node = ExportNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)


if __name__ == "__main__":
    unittest.main()
