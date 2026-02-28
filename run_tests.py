#!/usr/bin/env python
"""
All-to-Pipe test runner.

Run all tests with: python run_tests.py
Run specific test: python run_tests.py test_template_parser
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run the test suite."""
    # Discover and run all tests in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
