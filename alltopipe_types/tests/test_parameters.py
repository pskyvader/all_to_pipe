"""
Unit tests for Parameters type.

Tests the Parameters class and ParametersProcessor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types import Parameters


class TestParameters(unittest.TestCase):
    """Test suite for Parameters type."""

    def test_parameters_creation(self):
        """Test creating a Parameters instance."""
        params = Parameters(
            steps=20,
            cfg=7.5,
            sampler="euler",
            scheduler="karras",
            seed=12345,
        )
        self.assertEqual(params.steps, 20)
        self.assertEqual(params.cfg, 7.5)
        self.assertEqual(params.sampler, "euler")
        self.assertEqual(params.scheduler, "karras")
        self.assertEqual(params.seed, 12345)

    def test_parameters_values(self):
        """Test various parameter values."""
        params = Parameters(
            steps=100,
            cfg=15.0,
            sampler="dpmpp_2m_karras",
            scheduler="exponential",
            seed=0,
        )
        self.assertEqual(params.steps, 100)
        self.assertEqual(params.cfg, 15.0)
        self.assertIn("dpm", params.sampler.lower())

    def test_parameters_min_values(self):
        """Test Parameters with minimum values."""
        params = Parameters(
            steps=1,
            cfg=0.0,
            sampler="euler",
            scheduler="simple",
            seed=0,
        )
        self.assertEqual(params.steps, 1)
        self.assertEqual(params.cfg, 0.0)

    def test_parameters_max_values(self):
        """Test Parameters with maximum values."""
        params = Parameters(
            steps=100,
            cfg=20.0,
            sampler="euler",
            scheduler="karras",
            seed=0xFFFFFFFF,
        )
        self.assertEqual(params.steps, 100)
        self.assertEqual(params.cfg, 20.0)


if __name__ == "__main__":
    unittest.main()
