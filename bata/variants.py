"""Paper expert-sufficiency variants, not replacements for original BATA."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from . import core as c22
from .core import fit_bootstrap_ridge, score_ridge, ts_select
THRESHOLDS = {"g20": 0.20, "g30": 0.30, "g40": 0.40}

def ts_members(keys, scores, n, rng):
    """Same uniform-member TS as upstream, with M derived from the matrix.
    M=5 is delegated verbatim; larger M retains tie handling and uniqueness.
    """
    if scores.shape == (len(keys), 5):
        return ts_select(keys, scores, n, rng)
    assert scores.ndim == 2 and scores.shape[0] == len(keys) and (len(keys) >= n)
    assert scores.shape[1] > 0 and np.isfinite(scores).all()
    members = scores.shape[1]
    orders = [np.lexsort((keys, -scores[:, m])) for m in range(members)]
    pointers = [0] * members
    used = set()
    chosen = []
    for m in rng.integers(members, size=n):
        m = int(m)
        while int(orders[m][pointers[m]]) in used:
            pointers[m] += 1
        i = int(orders[m][pointers[m]])
        chosen.append((i, m))
        used.add(i)
    return chosen

def recipe(benchmark):
    if benchmark == 'GRB2':
        return ('plm', 'rf100')
    return c22.FINE_RECIPE[benchmark]

def ridge20(data, kind, train, y, ids, seed):
    x = np.asarray(data.feature(kind, train), np.float64)
    state = fit_bootstrap_ridge(x, y, seed, kind)
    xx = (x - state['mean']) / state['scale']
    yy = (np.asarray(y, np.float64) - state['ym']) / state['ys']
    rng = np.random.default_rng(seed)
    for _ in range(5):
        rng.integers(len(x), size=len(x))
    coef = list(state['coef'])
    inter = list(state['intercept'])
    for _ in range(5, 20):
        ii = rng.integers(len(x), size=len(x))
        m = Ridge(alpha=state['alpha']).fit(xx[ii], yy[ii])
        coef.append(m.coef_.astype(np.float64))
        inter.append(float(m.intercept_))
    state = dict(state, coef=np.stack(coef), intercept=np.asarray(inter))
    pred = np.empty((len(ids), 20), np.float64)
    for s in range(0, len(ids), 8192):
        kk = ids[s:s + 8192]
        pred[s:s + len(kk)] = score_ridge(state, data.feature(kind, kk))
    return (pred, dict(type='bootstrap_ridge20', kind=kind, alpha=float(state['alpha'])))

def rf20(data, kind, train, y, ids, algorithm_seed, rnd):
    x = data.feature(kind, train)
    pred = np.empty((len(ids), 20), np.float64)
    meta = []
    blocks = [(s, data.feature(kind, ids[s:s + 32768])) for s in range(0, len(ids), 32768)]
    for m in range(20):
        seed = int(np.random.SeedSequence([algorithm_seed, rnd, 5501, m]).generate_state(1)[0])
        boot = np.random.default_rng(seed).integers(len(train), size=len(train))
        model = RandomForestRegressor(n_estimators=100, max_features=1.0, bootstrap=True, n_jobs=1, random_state=seed).fit(x[boot], np.asarray(y)[boot])
        for (s, z) in blocks:
            pred[s:s + len(z), m] = model.predict(z)
        meta.append(dict(member=m, seed=seed, unique_train=int(len(np.unique(boot)))))
    return (pred, dict(type='bootstrap_rf20', kind=kind, trees_per_member=100, members=meta))

def fine_choose(benchmark, data, train, y, keys, fit_seed, rng, algorithm_seed, rnd):
    (kind, head) = recipe(benchmark)
    if head == 'ridge':
        (pred, meta) = ridge20(data, kind, train, y, keys, fit_seed)
    elif head in ('rf100', 'bootrf5'):
        (pred, meta) = rf20(data, kind, train, y, keys, algorithm_seed, rnd)
    else:
        raise ValueError((benchmark, kind, head))
    choices = ts_members(keys, pred, 96, rng)
    return (choices, keys, pred, dict(selection='fine_only_m20', selector='TS96', fine_representation=kind, fine_head=head, assay_model=meta, members=20))

def recipe_key(benchmark: str) -> str:
    return 'PABP' if benchmark == 'GRB2' else benchmark

def gate_choose(benchmark, arm, data, train, y, keys, fit_seed, rng, algorithm_seed, rnd):
    if arm == 'fine_m20':
        (ch, kept, pred, meta) = fine_choose(benchmark, data, train, y, keys, fit_seed, rng, algorithm_seed, rnd)
        return (ch, kept, pred, dict(meta, gate_arm=arm, gate_branch='fine_m20', gate_threshold=1.0))
    if arm not in THRESHOLDS:
        raise ValueError(arm)
    b = recipe_key(benchmark)
    (oe, of, foldmeta) = c22.crossfit_oof(data, b, train, y, fit_seed, algorithm_seed, rnd)
    w = c22.bata_weight(y, oe, of)
    tau = float(THRESHOLDS[arm])
    we = float(w['w_evolution'])
    if we < tau:
        (ch, kept, pred, meta) = fine_choose(benchmark, data, train, y, keys, fit_seed, rng, algorithm_seed, rnd)
        branch = 'fine_m20'
    else:
        (ch, pred, meta) = c22.rank_fused_select(data, b, train, y, keys, fit_seed, rng, algorithm_seed, rnd, we, 'expert_sufficiency_gate_fusion')
        kept = keys
        branch = 'bata_fusion'
    extra = dict(gate_arm=arm, gate_threshold=tau, gate_branch=branch, gate_w_evolution=we, gate_w_assay=1.0 - we, tail_arbitration=w, crossfit_folds=c22.NFOLD, oof_evolution_sha256=c22.ahash(oe), oof_assay_sha256=c22.ahash(of), folds=foldmeta)
    return (ch, kept, pred, dict(meta, **extra))
