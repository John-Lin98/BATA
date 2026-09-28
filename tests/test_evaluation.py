import unittest
from bata.evaluation import closed_loop


class TestEvaluation(unittest.TestCase):
    def test_complete_feedback_boundary_and_metric(self):
        seen_calls = []
        def select(rnd, observed):
            self.assertEqual(len(observed), 96 * rnd)
            self.assertEqual(set(observed), set(range(96 * rnd)))
            return list(range(96 * rnd, 96 * (rnd + 1)))
        def reveal(batch):
            seen_calls.append(batch)
            return {i: i / 480 for i in batch}
        result = closed_loop(list(range(96)), 480, select, reveal)
        self.assertEqual(len(seen_calls), 5)
        self.assertEqual(result['queries'], 480)
        history = [95/480, 191/480, 287/480, 383/480, 479/480]
        self.assertEqual(result['history'], history)
        self.assertEqual(result['Query_AUC'], (history[0]+2*sum(history[1:4])+history[4])/8)

    def test_invalid_selection_rejected_before_reveal(self):
        calls = []
        def reveal(batch):
            calls.append(batch)
            return {i: 0.0 for i in batch}
        with self.assertRaises(ValueError):
            closed_loop(list(range(96)), 480, lambda rnd, obs: list(range(96)), reveal)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
