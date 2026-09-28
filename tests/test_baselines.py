import unittest
import numpy as np
from bata.baselines import forest_select
from bata.core import fit_rf, score_blocks


class ForestRecipes(unittest.TestCase):
    def test_recipes_and_protocol(self):
        rng = np.random.default_rng(47)
        arrays = {kind: rng.normal(size=(240, 7)) for kind in ('plm', 'onehot')}
        class Data:
            def feature(self, kind, ids):
                return arrays[kind][ids]
        data = Data()
        train, keys, y = np.arange(96), np.arange(96, 240), rng.normal(size=96)
        for method, kind, authored in [('rf_onehot','onehot',False),
                                       ('evolvepro650','plm',True)]:
            chosen, scores, meta = forest_select(data, train, y, keys, 19, method)
            model = fit_rf(data.feature(kind, train), y, 19, authored=authored)
            expected = score_blocks(model, lambda ids: data.feature(kind, ids), keys)
            np.testing.assert_array_equal(scores[:, 0], expected)
            self.assertEqual(chosen, [(int(j), -1) for j in np.lexsort((keys, -expected))[:96]])
            self.assertEqual(meta['representation'], kind)
        with self.assertRaises(ValueError):
            forest_select(data, train, y, train, 19, 'rf_onehot')


if __name__ == '__main__':
    unittest.main()
