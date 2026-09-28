"""Local random-forest adaptations used in the frozen paper comparison.

These are recipe wrappers, not redistributed upstream method repositories.
The data provider exposes feature(kind, indices); only revealed y is accepted.
"""
import numpy as np
from .core import fit_rf, score_blocks


def forest_select(data, train, y, keys, fit_seed, method):
    if method not in ('rf_onehot', 'evolvepro650'):
        raise ValueError('Expected rf_onehot or evolvepro650')
    train, keys, y = np.asarray(train), np.asarray(keys), np.asarray(y)
    if (train.ndim != 1 or keys.ndim != 1 or y.shape != train.shape
            or len(np.unique(train)) != len(train)
            or len(np.unique(keys)) != len(keys) or len(keys) < 96
            or np.intersect1d(train, keys).size or not np.isfinite(y).all()):
        raise ValueError('Require distinct disjoint candidate IDs and revealed labels')
    authored = method == 'evolvepro650'
    kind = 'plm' if authored else 'onehot'
    getter = lambda ids: data.feature(kind, ids)
    model = fit_rf(getter(train), y, fit_seed, authored=authored)
    scores = score_blocks(model, getter, keys)
    chosen = [(int(j), -1) for j in np.lexsort((keys, -scores))[:96]]
    return chosen, scores[:, None], dict(representation=kind,
        fine_head='rf100' if authored else 'rf200')
