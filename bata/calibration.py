"""Matched calibration objectives; only the weighting objective varies."""
import numpy as np
from scipy.stats import rankdata
from . import core as c22
OBJECTIVES = ('G-Rank', 'T-Uniform-96', 'T-DCG-96', 'T-DCG-192')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def weighting(y, objective):
    """Average-rank tie handling exactly follows original BATA (no new tie rule)."""
    require(objective in OBJECTIVES, 'unknown calibration objective')
    n = len(y)
    if objective == 'G-Rank':
        return np.ones(n, np.float64)
    k = min(192 if objective == 'T-DCG-192' else 96, n)
    pos = rankdata(-np.asarray(y, np.float64), method='average')
    if objective == 'T-Uniform-96':
        return (pos <= k).astype(np.float64)
    return np.where(pos <= k, 1.0 / np.log2(pos + 1.0), 0.0)

def objective_weight(y, oe, of, objective):
    (y, oe, of) = [np.asarray(x, np.float64) for x in (y, oe, of)]
    require(y.ndim == 1 and y.shape == oe.shape == of.shape and (len(y) > 0), 'invalid objective inputs')
    require(all((np.isfinite(x).all() for x in (y, oe, of))), 'nonfinite input')
    if objective == 'T-DCG-96':
        return c22.bata_weight(y, oe, of)
    (target, re, rf) = [c22.percentile_rank(x) for x in (y, oe, of)]
    q = weighting(y, objective)
    (d, t) = (re - rf, target - rf)
    (den, num) = (float(np.sum(q * d * d)), float(np.sum(q * d * t)))
    w = 0.0 if den <= np.finfo(np.float64).eps else float(np.clip(num / den, 0.0, 1.0))
    return dict(w_evolution=w, w_assay=1.0 - w, k_tail=len(y) if objective == 'G-Rank' else min(len(y), 192 if objective == 'T-DCG-192' else 96), objective=objective, positive_weight_count=int((q > 0).sum()), oof_evolution_rank_sha256=c22.ahash(re), oof_assay_rank_sha256=c22.ahash(rf))

def unit_test():
    """Synthetic arrays only. Closed-form minima, ties, bounds and original parity."""
    rng = np.random.default_rng(20260916)
    count = 0
    for n in (2, 96, 192, 384):
        for tied in (False, True):
            y = rng.normal(size=n)
            if tied:
                y = np.round(y, 0)
            (oe, of) = (rng.normal(size=n), rng.normal(size=n))
            for objective in OBJECTIVES:
                out = objective_weight(y, oe, of, objective)
                w = out['w_evolution']
                (target, re, rf) = [c22.percentile_rank(x) for x in (y, oe, of)]
                q = weighting(y, objective)

                def loss(a):
                    return float(np.sum(q * (target - (a * re + (1.0 - a) * rf)) ** 2))
                require(0 <= w <= 1 and out['w_assay'] == 1 - w, 'invalid convex weight')
                require(loss(w) <= min((loss(a) for a in np.linspace(0, 1, 101))) + 1e-12, 'closed-form not minimum')
                if objective == 'T-DCG-96':
                    require(out == c22.bata_weight(y, oe, of), 'original weight not exact')
                if n <= 96:
                    np.testing.assert_array_equal(weighting(y, 'G-Rank'), weighting(y, 'T-Uniform-96'))
                    np.testing.assert_array_equal(weighting(y, 'T-DCG-96'), weighting(y, 'T-DCG-192'))
                if n > 1:
                    same = objective_weight(y, of, of, objective)
                    require(same['w_evolution'] == 0.0, 'degenerate denominator rule changed')
                count += 1
    return dict(valid=True, synthetic_cases=count, original_weight_direct_call=True, rank_conversion='average_rank/n', ties='original average-rank cutoff retained')
