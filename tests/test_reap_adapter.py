"""Wrapper contract test with fake training; does not execute a GPU experiment."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from bata.reap_adapter import reap_scores, reap_select


class REAPContract(unittest.TestCase):
    def test_100_members_and_acquisition_schedule(self):
        calls = []
        def train(**kwargs):
            self.assertEqual(kwargs['epochs'], 1000)
            self.assertEqual(kwargs['patience'], 50)
            self.assertEqual(kwargs['batch_size'], 1024)
            self.assertEqual(kwargs['device'], 'cuda')
            self.assertEqual(kwargs['model_type'], 'mlp')
            self.assertEqual((kwargs['alpha'], kwargs['margin']), (.8, .001))
            self.assertEqual((kwargs['lr'], kwargs['wd']), (3e-4, 1e-5))
            self.assertIsNone(kwargs['save_path'])
            calls.append(kwargs['seed'])
            return .2, .3, None, kwargs['seed']
        def predict(model, x, batch_size, device):
            self.assertEqual((batch_size, device), (4096, 'cuda'))
            return x[:, 0] + model % 7
        coefficients = []
        def rank(mean, std, *, lambda_sigma, top_n):
            coefficients.append(lambda_sigma)
            values = mean + lambda_sigma * std
            return np.argsort(-values)[:top_n], values
        modules = {'reap': types.ModuleType('reap'),
                   'reap.training': types.ModuleType('reap.training'),
                   'reap.selection': types.ModuleType('reap.selection')}
        modules['reap.training'].train_plm_rankreg = train
        modules['reap.training'].predict_plm_model = predict
        modules['reap.selection'].rank_by_ucb = rank
        features = np.random.default_rng(12).normal(size=(200, 5))
        class Data:
            def feature(self, kind, ids):
                assert kind == 'plm'
                return features[ids]
        train_ids, keys, y = np.arange(96), np.arange(96, 200), np.arange(96.)
        with patch.dict(sys.modules, modules):
            for rnd in (1, 4):
                choices, scores, meta = reap_select(Data(), train_ids, y, keys, 12, rnd)
                self.assertEqual(scores.shape, (104, 100))
                self.assertEqual(len(choices), 96)
                self.assertEqual(len(meta['members']), 100)
                for member in meta['members']:
                    self.assertFalse(set(member['train_ids']) & set(member['val_ids']))
                    self.assertEqual(set(member['train_ids']) | set(member['val_ids']), set(train_ids))
        self.assertEqual(len(calls), 200)
        self.assertEqual(coefficients, [.5, 0.])


if __name__ == '__main__':
    unittest.main()
