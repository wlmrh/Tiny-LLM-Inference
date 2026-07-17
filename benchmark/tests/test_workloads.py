import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suite.workloads import generate_arrival_ms


class ArrivalGenerationTest(unittest.TestCase):
    def test_simultaneous(self):
        self.assertEqual(generate_arrival_ms(4), [0.0, 0.0, 0.0, 0.0])

    def test_fixed(self):
        self.assertEqual(generate_arrival_ms(4, "fixed", 2.0), [0.0, 500.0, 1000.0, 1500.0])

    def test_poisson_is_seeded_and_monotonic(self):
        first = generate_arrival_ms(20, "poisson", 8.0, 1234)
        second = generate_arrival_ms(20, "poisson", 8.0, 1234)
        different = generate_arrival_ms(20, "poisson", 8.0, 5678)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(first[0], 0.0)
        self.assertTrue(all(lhs <= rhs for lhs, rhs in zip(first, first[1:])))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate_arrival_ms(-1)
        with self.assertRaises(ValueError):
            generate_arrival_ms(2, "fixed", 0.0)
        with self.assertRaises(ValueError):
            generate_arrival_ms(2, "unknown", 1.0)


if __name__ == "__main__":
    unittest.main()
