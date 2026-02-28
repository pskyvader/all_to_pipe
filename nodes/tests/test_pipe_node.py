"""
Unit tests for PipeNode.

Tests the Pipe creation node.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nodes.pipe_node import PipeNode
from alltopipe_types import Pipe


class TestPipeNode(unittest.TestCase):
    """Test suite for PipeNode."""

    def test_pipe_node_execution(self):
        """Test PipeNode creates an empty Pipe."""
        result = PipeNode.execute()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

        pipe = result[0]
        self.assertIsInstance(pipe, Pipe)
        self.assertIsNone(pipe.model)
        self.assertEqual(pipe.loras, [])
        self.assertIsNone(pipe.parameters)
        self.assertIsNone(pipe.image_config)

    def test_pipe_node_prompts_initialized(self):
        """Test that PipeNode initializes prompts."""
        result = PipeNode.execute()
        pipe = result[0]

        self.assertIsNotNone(pipe.positive_prompt)
        self.assertIsNotNone(pipe.negative_prompt)


if __name__ == "__main__":
    unittest.main()
