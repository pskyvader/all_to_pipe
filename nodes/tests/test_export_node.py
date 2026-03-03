"""
Unit tests for ExportNode.

Tests the sampler export node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..export_node import ExportNode
from ...alltopipe_types import Pipe, Model, Parameters


class TestExportNode(unittest.TestCase):
    """Test suite for ExportNode."""

    def test_node_class_attributes(self):
        """Test ExportNode class attributes and verify all parameters are exported."""
        expected_returns = (
            "MODEL",
            "CONDITIONING",
            "CONDITIONING",
            "INT",
            "INT",
            "FLOAT",
            "STRING",
            "STRING",
            "INT",
            "INT",
            "INT",
            "FLOAT",
        )
        self.assertEqual(ExportNode.RETURN_TYPES, expected_returns)
        self.assertEqual(len(ExportNode.RETURN_NAMES), 12)
        self.assertEqual(ExportNode.FUNCTION, "execute")
        self.assertEqual(ExportNode.CATEGORY, "all-to-pipe")

    def test_node_has_required_inputs(self):
        """Test that ExportNode has required inputs."""
        node = ExportNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)


if __name__ == "__main__":
    unittest.main()
