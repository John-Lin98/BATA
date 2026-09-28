import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch not installed')
class DCADecomposition(unittest.TestCase):
    def test_features_match_direct_energy(self):
        from bata.dca_features import _features
        rng = np.random.default_rng(18)
        for width in (4, 7):
            codes = rng.integers(1, 21, size=(9, width))
            fields = rng.normal(size=(21, width))
            couplings = rng.normal(size=(width, width, 21, 21))
            expected = np.array([[fields[z[i], i] + .5 * sum(
                couplings[i, j, z[i], z[j]] for j in range(width) if j != i)
                for i in range(width)] for z in codes], dtype=np.float32)
            actual, _ = _features(codes, fields, couplings, batch=3)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':unittest.main()
