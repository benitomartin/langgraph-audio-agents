import unittest

def test_smoke() -> None:
    """A simple smoke test to check the test setup works."""
    unittest.TestCase().assertEqual(1 + 1, 2)
