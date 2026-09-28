"""Frozen ALDE five-DNN adaptation; upstream class licensed under MIT.

DNN_FF copyright (c) 2025 Jason Yang; see licenses/ALDE-MIT.txt.
Only observed feedback may be passed to fitting. No upstream data is bundled.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class DNN_FF(torch.nn.Sequential):
    """
    Standard DNN, feedforward.
    For use in DKL and DNN_ENSEMBLE
    """
    act_dict = {'relu': torch.nn.ReLU(), 'lrelu': torch.nn.LeakyReLU(), 'swish': torch.nn.SiLU(), 'sigmoid': torch.nn.Sigmoid(), 'tanh': torch.nn.Tanh(), 'softmax': torch.nn.Softmax()}

    def __init__(self, architecture, activation='relu', p_dropout=0, inference_args=None, device='cuda', *_, **__):
        super().__init__()
        self.device = device
        self.architecture = architecture
        act_layer = self.act_dict[activation.lower()]
        self.inference_args = inference_args
        self.dkl = True
        for dim in range(len(architecture)):
            name = str(dim + 1)
            if dim + 1 < len(architecture):
                self.add_module('linear' + name, torch.nn.Linear(architecture[dim], architecture[dim + 1]).double())
            if dim + 2 < len(architecture):
                if p_dropout > 0 and p_dropout < 1:
                    self.add_module('dropout' + name, torch.nn.Dropout(p=p_dropout))
                name = activation + name
                self.add_module(name, act_layer)

    def get_params(self):
        return [{'params': self.parameters()}]

    def train_model(self, X, Y, lr, num_iter=100, verbose=2, *_, **__):
        self.train()
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        mse = torch.nn.MSELoss()
        losses = np.zeros(num_iter)
        w = 30
        for i in range(num_iter):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = mse(preds, Y)
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()
            if i > w:
                recent_min = losses[i - w + 1:i + 1].min()
                overall_min = losses[:i - w + 1].min()
                if overall_min <= recent_min:
                    print('Early stopping at iteration ' + str(i))
                    break
        self.eval()
        return (self, None)

def fit(x, y):
    xt = torch.as_tensor(x, dtype=torch.float64)
    scale = float(np.max(y))
    assert scale > 0
    yt = torch.as_tensor(y / scale, dtype=torch.float64).reshape(-1, 1)
    states = []
    members = []
    for i in range(5):
        (ids, held) = train_test_split(np.arange(len(x)), test_size=1 - 0.9, random_state=1 + i)
        model = DNN_FF([x.shape[1], 30, 30, 1], activation='lrelu', p_dropout=0, device='cpu').double()
        calls = []
        hook = model.linear1.register_forward_hook(lambda module, args, result: calls.append(1))
        model.train_model(xt[ids], yt[ids], lr=0.001, num_iter=300)
        hook.remove()
        assert 32 <= len(calls) <= 300 and all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        assert any((torch.count_nonzero(p.grad) > 0 for p in model.parameters()))
        states.append({k: v.detach().clone() for (k, v) in model.state_dict().items()})
        members.append(dict(member=i, train_ids=ids.tolist(), held_ids=held.tolist(), actual_updates=len(calls)))
    return (dict(feature_fit_n=len(x), members=members, updates_cap=300, sampling='sklearn_split', scale=scale, precision='float64', holdout_used_for_stopping=False), dict(states=states, input_dim=x.shape[1], scale=scale))

def score_dnn(state, x, device='cpu', chunk_size=8192):
    """Vectorized original linear/lrelu layers; return real-scale [N,5]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != state['input_dim'] or (not np.isfinite(x).all()) or (len(state['states']) != 5) or (not np.isfinite(state['scale'])) or (state['scale'] <= 0) or (not isinstance(chunk_size, int)) or (chunk_size < 1):
        raise ValueError('invalid DNN scoring state/input/chunk size')
    result = np.empty((len(x), 5), dtype=np.float64)
    with torch.inference_mode():
        weights = {name: torch.stack([member[name] for member in state['states']]).to(device=device, dtype=torch.float64) for name in state['states'][0]}
        for start in range(0, len(x), chunk_size):
            values = torch.as_tensor(x[start:start + chunk_size], dtype=torch.float64, device=device).unsqueeze(0).expand(5, -1, -1)
            for layer in range(1, 4):
                values = torch.bmm(values, weights[f'linear{layer}.weight'].transpose(1, 2)) + weights[f'linear{layer}.bias'][:, None, :]
                if layer < 3:
                    values = torch.nn.functional.leaky_relu(values, negative_slope=0.01)
            result[start:start + chunk_size] = values.squeeze(-1).T.cpu().numpy() * state['scale']
    if not np.isfinite(result).all():
        raise RuntimeError('nonfinite DNN scores')
    return result

def fit_dnn(x, y, seed):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or len(x) < 10 or not x.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all() or y.max() <= 0):
        raise ValueError('Require finite observed x/y, >=10 rows and positive max(y)')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed) % (2 ** 63))
        meta, state = fit(x, y)
    meta['adapter_seed'] = int(seed)
    return meta, state


def alde_select(data, train, y, keys, fit_seed, rng):
    from .core import ts_select
    meta, model = fit_dnn(data.feature('onehot', train), y, fit_seed)
    scores = np.empty((len(keys), 5), dtype=np.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for start in range(0, len(keys), 16384):
        kk = keys[start:start + 16384]
        scores[start:start + len(kk)] = score_dnn(model,
            data.feature('onehot', kk), device=device, chunk_size=16384)
    return ts_select(keys, scores, 96, rng), scores, dict(fit=meta, score_device=device)
