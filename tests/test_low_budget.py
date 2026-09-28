import unittest
import numpy as np
from bata.low_budget import bata_weight16


class TestLowBudget(unittest.TestCase):
    def test_identical_and_perfect_experts(self):
        y = np.arange(32, dtype=float)
        result = bata_weight16(y, y, y)
        self.assertEqual(result['k_tail'], 16)
        self.assertEqual(result['w_evolution'], 0.0)
        self.assertEqual(bata_weight16(y, y, -y)['w_evolution'], 1.0)
        self.assertEqual(bata_weight16(y, -y, y)['w_evolution'], 0.0)


if __name__ == '__main__':
    unittest.main()
