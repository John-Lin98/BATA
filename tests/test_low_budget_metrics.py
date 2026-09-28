import unittest
import numpy as np
from scripts.evaluate_low_budget import endpoint_metrics


class TestLowBudgetMetrics(unittest.TestCase):
    def test_world_percentile_and_rejection(self):
        truth = np.arange(100, dtype=float)
        batches = [list(range(i, i+16)) for i in (52, 68, 84)]
        result = endpoint_metrics(truth, list(range(100)), batches)
        self.assertEqual(result['top10'], 11)  # pct >= .90, including the boundary
        self.assertTrue(result['hit1'])
        self.assertEqual(result['normalized_regret'], 0.0)
        with self.assertRaises(ValueError):
            endpoint_metrics(truth, list(range(100)), [batches[0]]*3)


if __name__ == '__main__':
    unittest.main()
