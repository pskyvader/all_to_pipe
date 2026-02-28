"""
Unit tests for constants module.

Tests the constants definitions including samplers and schedulers.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.constants import SUPPORTED_SAMPLERS, SUPPORTED_SCHEDULERS


class TestConstants(unittest.TestCase):
    """Test suite for constants."""

    def test_samplers_not_empty(self):
        """Test that SUPPORTED_SAMPLERS is populated."""
        self.assertGreater(len(SUPPORTED_SAMPLERS), 0)
        self.assertGreater(len(SUPPORTED_SAMPLERS), 40)  # Should have many

    def test_samplers_are_strings(self):
        """Test that all samplers are strings."""
        for sampler in SUPPORTED_SAMPLERS:
            self.assertIsInstance(sampler, str)
            self.assertGreater(len(sampler), 0)

    def test_common_samplers_present(self):
        """Test that common samplers are defined."""
        common = ["euler", "dpm_2", "heun"]
        for sampler in common:
            self.assertIn(sampler, SUPPORTED_SAMPLERS)

    def test_schedulers_not_empty(self):
        """Test that SUPPORTED_SCHEDULERS is populated."""
        self.assertEqual(len(SUPPORTED_SCHEDULERS), 9)

    def test_schedulers_are_strings(self):
        """Test that all schedulers are strings."""
        for scheduler in SUPPORTED_SCHEDULERS:
            self.assertIsInstance(scheduler, str)
            self.assertGreater(len(scheduler), 0)

    def test_common_schedulers_present(self):
        """Test that common schedulers are defined."""
        common = ["karras", "exponential", "simple"]
        for scheduler in common:
            self.assertIn(scheduler, SUPPORTED_SCHEDULERS)


if __name__ == "__main__":
    unittest.main()
