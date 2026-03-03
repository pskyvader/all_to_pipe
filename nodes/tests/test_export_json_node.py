"""
Unit tests for ExportJsonNode.

Tests the JSON export node.
"""

import sys
import unittest
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..export_json_node import ExportJsonNode
from ...alltopipe_types import Pipe, Model, Parameters, LoraSpec


class TestExportJsonNode(unittest.TestCase):
    """Test suite for ExportJsonNode."""

    def test_node_class_attributes(self):
        """Test ExportJsonNode class attributes."""
        self.assertEqual(ExportJsonNode.RETURN_TYPES, ("STRING",))
        self.assertEqual(ExportJsonNode.RETURN_NAMES, ("json",))
        self.assertEqual(ExportJsonNode.FUNCTION, "execute")
        self.assertEqual(ExportJsonNode.CATEGORY, "all-to-pipe")

    def test_node_has_required_inputs(self):
        """Test that ExportJsonNode has required inputs."""
        node = ExportJsonNode()
        inputs = node.INPUT_TYPES()

        required = inputs.get("required", {})
        self.assertIn("pipe", required)

    def test_node_returns_valid_json(self):
        """Test that ExportJsonNode returns valid JSON with new flat structure."""
        # Create a minimal pipe
        pipe = Pipe(
            model=Model(name="test.safetensors", subfolder="checkpoints"),
            parameters=Parameters(
                steps=20,
                cfg=7.5,
                sampler="euler",
                scheduler="karras",
                seed=12345,
            ),
        )

        # Execute should return a tuple with JSON string
        result = ExportJsonNode.execute(pipe)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

        json_string = result[0]
        self.assertIsInstance(json_string, str)

        # Verify it's valid JSON
        try:
            parsed = json.loads(json_string)
            self.assertIsInstance(parsed, dict)
            
            # Verify new flat consolidated structure
            # Properties should be at top level, not nested
            self.assertIn("model", parsed)
            self.assertIn("model_subfolder", parsed)
            self.assertIn("steps", parsed)
            self.assertIn("cfg", parsed)
            self.assertIn("sampler", parsed)
            self.assertIn("scheduler", parsed)
            self.assertIn("seed", parsed)
            
            # Verify values are correct
            self.assertEqual(parsed["model"], "test.safetensors")
            self.assertEqual(parsed["model_subfolder"], "checkpoints")
            self.assertEqual(parsed["steps"], 20)
            self.assertEqual(parsed["cfg"], 7.5)
            self.assertEqual(parsed["sampler"], "euler")
            self.assertEqual(parsed["scheduler"], "karras")
            self.assertEqual(parsed["seed"], 12345)
            
        except json.JSONDecodeError:
            self.fail("ExportJsonNode did not return valid JSON")


if __name__ == "__main__":
    unittest.main()
