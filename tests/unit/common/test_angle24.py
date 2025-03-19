import math
import unittest

from common import angle24


class Angle24TestCase(unittest.TestCase):
    def test_cyclic(self):
        """Verify that encoding a value that was decoded will always give the same result."""
        decoded = [angle24.decode(i) for i in range(0xFFFFFF)]
        encoded = [angle24.encode(d) for d in decoded]
        self.assertTrue(all(i == e for i, e in enumerate(encoded)))

    def test_rounding(self):
        """Verify that the rounding error remains below a certain threshold."""
        self.assertEqual(-math.pi, angle24.decode(angle24.encode(-math.pi)))
        self.assertEqual(0.0, angle24.decode(angle24.encode(0.0)))
        self.assertEqual(-math.pi, angle24.decode(angle24.encode(math.pi)))

        max_error = math.pi / 0xFFFFFF  # Error should be at most half the interval size
        increment = 0.001
        cases = [-math.pi + increment * i for i in range(int(2 * math.pi / increment))]
        errors = [val - angle24.decode(angle24.encode(val)) for val in cases]
        self.assertTrue(all(error <= max_error for error in errors))
