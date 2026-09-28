import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch dependency not installed')
class ALDEAdapter(unittest.TestCase):
    def test_seed_scaling_and_vectorized_score(self):
        import torch
        from bata.alde import fit_dnn, score_dnn, DNN_FF
        torch.set_num_threads(1)
        x = np.random.default_rng(12).normal(size=(96, 8))
        y = np.linspace(.1, 1, 96)
        before = torch.random.get_rng_state().clone()
        meta, state = fit_dnn(x, y, 38)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(len(meta['members']), 5)
        direct = []
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            net = DNN_FF([8, 30, 30, 1], activation='lrelu', device='cpu').double().eval()
            for weights in state['states']:
                net.load_state_dict(weights)
                direct.append(net(torch.as_tensor(x)).numpy().ravel() * state['scale'])
        np.testing.assert_allclose(score_dnn(state, x), np.stack(direct, axis=1), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            fit_dnn(x, -np.ones(96), 38)


if __name__ == '__main__':
    unittest.main()
