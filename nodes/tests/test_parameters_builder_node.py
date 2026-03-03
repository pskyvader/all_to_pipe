"""
Unit tests for ParametersBuilderNode.

Tests the sampler parameters configuration node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..parameters_builder_node import ParametersBuilderNode
from ...alltopipe_types import Pipe, Parameters


class TestParametersBuilderNode(unittest.TestCase):
    """Test suite for ParametersBuilderNode."""

    def test_node_class_attributes(self):
        """Test ParametersBuilderNode class attributes."""
        self.assertEqual(ParametersBuilderNode.RETURN_TYPES, ("PIPE",))
        self.assertEqual(ParametersBuilderNode.RETURN_NAMES, ("pipe",))
        self.assertEqual(ParametersBuilderNode.FUNCTION, "execute")
        self.assertEqual(ParametersBuilderNode.CATEGORY, "all-to-pipe")

    def test_node_has_required_inputs(self):
        """Test that ParametersBuilderNode has optional pipe and required parameters."""
        node = ParametersBuilderNode()
        inputs = node.INPUT_TYPES()

        optional = inputs.get("optional", {})
        self.assertIn("pipe", optional)
        
        required = inputs.get("required", {})
        self.assertIn("steps", required)
        self.assertIn("cfg", required)
        self.assertIn("sampler", required)
        self.assertIn("scheduler", required)
        self.assertIn("seed", required)

    def test_sampler_and_scheduler_combo(self):
        """Test that sampler and scheduler have COMBO inputs."""
        node = ParametersBuilderNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        sampler = required.get("sampler")
        scheduler = required.get("scheduler")

        # Should have COMBO type
        self.assertIsNotNone(sampler)
        self.assertIsNotNone(scheduler)

    def test_node_step_and_cfg_ranges(self):
        """Test step and cfg input specifications."""
        node = ParametersBuilderNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        steps = required.get("steps", {})
        cfg = required.get("cfg", {})

        # Both should be present
        self.assertIsNotNone(steps)
        self.assertIsNotNone(cfg)


if __name__ == "__main__":
    unittest.main()
