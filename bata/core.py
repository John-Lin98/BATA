"""Frozen BATA M5 numerical kernel. Receives revealed labels only."""
import hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 4, 17)
ENSEMBLE = 5
NFOLD = 5
BATCH = 96
FINE_RECIPE = {'GB1': ('onehot','ridge'), 'PABP': ('plm','rf100'),
               'TrpB': ('plm','bootrf5'), 'HIS7': ('plm','ridge')}


def fit_bootstrap_ridge(x, y, seed, kind):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    scaler = StandardScaler().fit(x)
    xt = scaler.transform(x)
    ym = float(y.mean())
    ys = float(y.std(ddof=0))
    if ys <= 0:
        raise ValueError('constant y')
    z = (y - ym) / ys
    base = RidgeCV(alphas=ALPHAS, cv=None, scoring='neg_mean_squared_error').fit(xt, z)
    alpha = float(base.alpha_)
    rng = np.random.default_rng(seed)
    coefs = []
    inter = []
    for m in range(ENSEMBLE):
        b = rng.integers(len(x), size=len(x))
        model = Ridge(alpha=alpha).fit(xt[b], z[b])
        coefs.append(model.coef_.astype(np.float64))
        inter.append(float(model.intercept_))
    state = dict(kind=kind, alpha=alpha, mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64), coef=np.stack(coefs), intercept=np.asarray(inter), ym=ym, ys=ys)
    return state

def score_ridge(state, x):
    x = np.asarray(x, np.float64)
    xs = (x - state['mean']) / state['scale']
    pred = xs @ state['coef'].T + state['intercept']
    return pred * state['ys'] + state['ym']

def ts_select(keys, scores, n, rng):
    assert len(keys) >= n and scores.shape == (len(keys), 5) and np.isfinite(scores).all()
    orders = [np.lexsort((keys, -scores[:, member])) for member in range(5)]
    (used, chosen, pointers) = (set(), [], [0] * 5)
    for member in rng.integers(5, size=n):
        member = int(member)
        while int(orders[member][pointers[member]]) in used:
            pointers[member] += 1
        j = int(orders[member][pointers[member]])
        chosen.append((j, member))
        used.add(j)
    return chosen

def fit_rf(x, y, seed, authored=False):
    kwargs = dict(n_estimators=100 if authored else 200, max_features=1.0, bootstrap=True, n_jobs=1, random_state=1 if authored else seed)
    if authored:
        kwargs['criterion'] = 'friedman_mse'
    return RandomForestRegressor(**kwargs).fit(x, y)

def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def score_blocks(model, getter, keys, ridge=False):
    pred = np.empty((len(keys), 5) if ridge else len(keys), dtype=np.float64)
    for start in range(0, len(keys), 8192):
        block = keys[start:start + 8192]
        x = getter(block)
        pred[start:start + len(block)] = score_ridge(model, x) if ridge else model.predict(x)
    return pred

def _kind_getter(data, kind):
    return lambda ids: data.feature(kind, ids)

def _fit_predict_ridge(data, kind, train, y, keys, seed):
    getter = _kind_getter(data, kind)
    state = fit_bootstrap_ridge(getter(train), y, seed, kind)
    scores = np.empty((len(keys), 5), dtype=np.float64)
    for s in range(0, len(keys), 8192):
        kk = keys[s:s + 8192]
        scores[s:s + len(kk)] = score_ridge(state, getter(kk))
    return (scores, dict(type='ridge5', alpha=float(state['alpha'])))

def _fit_predict_rf100(data, kind, train, y, keys, seed):
    getter = _kind_getter(data, kind)
    model = fit_rf(getter(train), y, seed, authored=True)
    pred = score_blocks(model, getter, keys)[:, None]
    return (pred, dict(type='rf100_authored'))

def _fit_predict_single_rf_member(data, kind, train, y, keys, seed):
    """One RF100 member, used only inside cross-fit evidence estimation."""
    getter = _kind_getter(data, kind)
    model = RandomForestRegressor(n_estimators=100, max_features=1.0, bootstrap=True, n_jobs=1, random_state=int(seed)).fit(getter(train), y)
    pred = score_blocks(model, getter, keys)[:, None]
    return (pred, dict(type='rf100_single_crossfit', seed=int(seed)))

def _fit_predict_bootrf5(data, kind, train, y, keys, algorithm_seed, rnd, fold_tag=None):
    getter = _kind_getter(data, kind)
    (preds, meta) = ([], [])
    for member in range(5):
        if fold_tag is None:
            seed_seq = [algorithm_seed, rnd, 5501, member]
        else:
            seed_seq = [algorithm_seed, rnd, 5501, 2101, int(fold_tag), member]
        seed = int(np.random.SeedSequence(seed_seq).generate_state(1)[0])
        brng = np.random.default_rng(seed)
        boot = brng.integers(len(train), size=len(train))
        model = RandomForestRegressor(n_estimators=100, max_features=1.0, bootstrap=True, n_jobs=1, random_state=seed).fit(getter(train[boot]), y[boot])
        preds.append(score_blocks(model, getter, keys))
        meta.append(dict(member=member, seed=seed, unique_train=int(len(np.unique(boot)))))
    return (np.stack(preds, axis=1), dict(type='bootstrap_rf5', members=meta))

def expert_members(data, benchmark, expert, train, y, keys, seed, algorithm_seed, rnd, fold_tag=None):
    """Return [N,M] member predictions for one expert.

    Evolution expert is always DCA Ridge5. Assay expert exactly follows the
    benchmark-specific frozen fine recipe. Cross-fit models use fold-specific
    seeds; full models preserve the historical full-fit seed recipe.
    """
    if expert == 'evolution':
        use_seed = seed if fold_tag is None else int(np.random.SeedSequence([seed, 3101, int(fold_tag)]).generate_state(1)[0])
        return _fit_predict_ridge(data, 'dca', train, y, keys, use_seed)
    if expert != 'assay':
        raise ValueError(expert)
    (kind, head) = FINE_RECIPE[benchmark]
    use_seed = seed if fold_tag is None else int(np.random.SeedSequence([seed, 3201, int(fold_tag)]).generate_state(1)[0])
    if head == 'ridge':
        return _fit_predict_ridge(data, kind, train, y, keys, use_seed)
    if head == 'rf100':
        return _fit_predict_rf100(data, kind, train, y, keys, use_seed)
    if head == 'bootrf5':
        if fold_tag is not None:
            seed = int(np.random.SeedSequence([algorithm_seed, rnd, 5501, 2101, int(fold_tag)]).generate_state(1)[0])
            return _fit_predict_single_rf_member(data, kind, train, y, keys, seed)
        return _fit_predict_bootrf5(data, kind, train, y, keys, algorithm_seed, rnd, fold_tag=None)
    raise ValueError((benchmark, kind, head))

def percentile_rank(x):
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, method='average') / len(x)

def _assay_fold_predict(data, benchmark, train, y, val_ids, fit_seed, algorithm_seed, rnd, fold):
    (kind, head) = FINE_RECIPE[benchmark]
    if head == 'bootrf5':
        return _fit_predict_bootrf5(data, kind, train, y, val_ids, algorithm_seed, rnd, fold_tag=fold)
    return expert_members(data, benchmark, 'assay', train, y, val_ids, fit_seed, algorithm_seed, rnd, fold_tag=fold)

def crossfit_oof(data, benchmark, train, y, fit_seed, algorithm_seed, rnd):
    n = len(train)
    frng = np.random.default_rng(np.random.SeedSequence([algorithm_seed, rnd, 22022]))
    order = frng.permutation(n)
    folds = [np.asarray(z, dtype=np.int64) for z in np.array_split(order, NFOLD)]
    oe = np.empty(n, np.float64)
    of = np.empty(n, np.float64)
    meta = []
    local = np.arange(n, dtype=np.int64)
    for (k, val) in enumerate(folds):
        fit = np.setdiff1d(local, val, assume_unique=True)
        tr = train[fit]
        yy = y[fit]
        vi = train[val]
        (ep, em) = expert_members(data, benchmark, 'evolution', tr, yy, vi, fit_seed, algorithm_seed, rnd, fold_tag=k)
        (fp, fm) = _assay_fold_predict(data, benchmark, tr, yy, vi, fit_seed, algorithm_seed, rnd, k)
        oe[val] = ep.mean(1)
        of[val] = fp.mean(1)
        meta.append(dict(fold=k, n_train=len(fit), n_val=len(val), evolution=em, assay=fm))
    assert np.isfinite(oe).all() and np.isfinite(of).all()
    return (oe, of, meta)

def bata_weight(y, oof_e, oof_f):
    """Closed-form top-B discounted rank regression weight.

    True and predicted values are converted to percentile ranks. Only the
    observed top min(B,n) fitness points receive weight, with standard DCG
    discount 1/log2(position+1). The only cutoff is the actual assay batch B.
    """
    y = np.asarray(y, np.float64)
    n = len(y)
    k = min(BATCH, n)
    target = percentile_rank(y)
    re = percentile_rank(oof_e)
    rf = percentile_rank(oof_f)
    pos = rankdata(-y, method='average')
    discount = np.where(pos <= k, 1.0 / np.log2(pos + 1.0), 0.0)
    d = re - rf
    t = target - rf
    den = float(np.sum(discount * d * d))
    num = float(np.sum(discount * d * t))
    we = 0.0 if den <= np.finfo(np.float64).eps else float(np.clip(num / den, 0.0, 1.0))
    comb = we * re + (1 - we) * rf

    def risk(pred):
        z = target - pred
        return float(np.sum(discount * z * z) / max(float(discount.sum()), np.finfo(float).eps))
    nd = dict(evolution=float(ndcg_score(target[None], re[None], k=k)), assay=float(ndcg_score(target[None], rf[None], k=k)), combined=float(ndcg_score(target[None], comb[None], k=k)))
    return dict(w_evolution=we, w_assay=1.0 - we, k_tail=int(k), tail_rank_risk_evolution=risk(re), tail_rank_risk_assay=risk(rf), tail_rank_risk_combined=risk(comb), ndcg_at_batch=nd, oof_evolution_rank_sha256=ahash(re), oof_assay_rank_sha256=ahash(rf))

def rank_members(scores):
    scores = np.asarray(scores, np.float64)
    out = np.empty_like(scores)
    for m in range(scores.shape[1]):
        out[:, m] = percentile_rank(scores[:, m])
    return out

def align_members(e, f):
    m = max(e.shape[1], f.shape[1])
    if e.shape[1] == 1 and m > 1:
        e = np.repeat(e, m, axis=1)
    if f.shape[1] == 1 and m > 1:
        f = np.repeat(f, m, axis=1)
    if e.shape[1] != f.shape[1]:
        raise RuntimeError('expert member count mismatch')
    return (e, f)

def rank_fused_select(data, benchmark, train, y, keys, fit_seed, rng, algorithm_seed, rnd, we, tag):
    (ep, em) = expert_members(data, benchmark, 'evolution', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    (fp, fm) = expert_members(data, benchmark, 'assay', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    (er, fr) = align_members(rank_members(ep), rank_members(fp))
    combined = we * er + (1 - we) * fr
    (kind, head) = FINE_RECIPE[benchmark]
    if head == 'rf100':
        mean = combined.mean(1)
        order = np.lexsort((keys, -mean))[:BATCH]
        ch = [(int(j), -1) for j in order]
        selector = 'greedy'
    else:
        ch = ts_select(keys, combined, BATCH, rng)
        selector = 'TS96'
    return (ch, combined, dict(selection=tag, w_evolution=float(we), w_assay=float(1 - we), score_space='candidate_percentile_rank', selector=selector, fine_representation=kind, fine_head=head, evolution_model=em, assay_model=fm))
