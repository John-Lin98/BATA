import unittest
import numpy as np
from unittest.mock import patch
from bata import controls


class TestControls(unittest.TestCase):
    def test_weights_and_frozen_fold_seed(self):
        y = np.arange(96, dtype=float)
        weights = controls.min_variance_weights(y, y, y)
        self.assertEqual(weights['w_evolution'], 0.5)
        records = []
        def expert(data, benchmark, kind, train, values, keys, seed, algorithm_seed, rnd, fold_tag):
            records.append((kind, fold_tag, keys.copy()))
            return np.repeat(keys[:, None].astype(float), 5, axis=1), {}
        with patch.object(controls, 'expert_members', expert):
            evidence = controls.crossfit_evidence(None, 'GB1', np.arange(96), y,
                                                  np.arange(96, 200), 1, 42, 2, False)
        folds = np.array_split(np.random.default_rng(np.random.SeedSequence([42, 2, 9917])).permutation(96), 5)
        for k, fold in enumerate(folds):
            np.testing.assert_array_equal(records[2*k][2], fold)
        np.testing.assert_array_equal(evidence['oof_e'], y)
        np.testing.assert_array_equal(evidence['oof_f'], y)
        self.assertIsNone(evidence['cand_e'])


if __name__ == '__main__':
    unittest.main()
