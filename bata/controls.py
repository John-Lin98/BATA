"""Frozen paper controls: preserve their original OOF seeds and score spaces."""
import numpy as np
from sklearn.covariance import LedoitWolf
from .core import FINE_RECIPE, expert_members, ts_select, rank_members
BATCH = 96
NFOLD = 5


def crossfit_evidence(data, benchmark, train, y, keys, fit_seed, algorithm_seed, rnd, need_candidate_envelopes=True):
    """Cross-fitted evidence for safe screening and expert arbitration.

    Each fold model predicts its held-out training fold and all current
    candidates. Fold-to-fold candidate variation acts as a model-evidence
    envelope; no full-landscape labels are read.
    """
    n = len(train)
    if n < 20:
        raise ValueError('cross-fit requires >=20 observed labels')
    fold_rng = np.random.default_rng(np.random.SeedSequence([algorithm_seed, rnd, 9917]))
    order = fold_rng.permutation(n)
    folds = [np.asarray(x, dtype=np.int64) for x in np.array_split(order, NFOLD)]
    oof_e = np.empty(n, dtype=np.float64)
    oof_f = np.empty(n, dtype=np.float64)
    fold_id = np.empty(n, dtype=np.int64)
    cand_e = np.empty((len(keys), NFOLD), dtype=np.float64) if need_candidate_envelopes else None
    cand_f = np.empty((len(keys), NFOLD), dtype=np.float64) if need_candidate_envelopes else None
    fold_meta = []
    all_local = np.arange(n, dtype=np.int64)
    for (k, val_local) in enumerate(folds):
        fit_local = np.setdiff1d(all_local, val_local, assume_unique=True)
        tr_ids = train[fit_local]
        yy = y[fit_local]
        val_ids = train[val_local]
        (e_val, emeta) = expert_members(data, benchmark, 'evolution', tr_ids, yy, val_ids, fit_seed, algorithm_seed, rnd, fold_tag=k)
        (f_val, fmeta) = expert_members(data, benchmark, 'assay', tr_ids, yy, val_ids, fit_seed, algorithm_seed, rnd, fold_tag=k)
        oof_e[val_local] = e_val.mean(axis=1)
        oof_f[val_local] = f_val.mean(axis=1)
        fold_id[val_local] = k
        if need_candidate_envelopes:
            (e_cand, _) = expert_members(data, benchmark, 'evolution', tr_ids, yy, keys, fit_seed, algorithm_seed, rnd, fold_tag=k)
            (f_cand, _) = expert_members(data, benchmark, 'assay', tr_ids, yy, keys, fit_seed, algorithm_seed, rnd, fold_tag=k)
            cand_e[:, k] = e_cand.mean(axis=1)
            cand_f[:, k] = f_cand.mean(axis=1)
        fold_meta.append(dict(fold=k, n_train=len(fit_local), n_val=len(val_local), evolution=emeta, assay=fmeta))
    assert np.isfinite(oof_e).all() and np.isfinite(oof_f).all()
    if need_candidate_envelopes:
        assert np.isfinite(cand_e).all() and np.isfinite(cand_f).all()
    return dict(oof_e=oof_e, oof_f=oof_f, cand_e=cand_e, cand_f=cand_f, fold_id=fold_id, folds=fold_meta)

def min_variance_weights(y, oof_e, oof_f):
    """Two-expert minimum-variance weights from cross-fitted residuals.

    Each expert is first OOF-debiased. Then w minimizes residual variance under
    w in [0,1], w_e + w_f = 1. No hand-tuned mixing coefficient is used.
    """
    bias_e = float(np.mean(y - oof_e))
    bias_f = float(np.mean(y - oof_f))
    err_e = y - (oof_e + bias_e)
    err_f = y - (oof_f + bias_f)
    errors = np.stack([err_e, err_f], axis=1)
    cov = LedoitWolf(assume_centered=True).fit(errors).covariance_
    (see, sff, sef) = (float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1]))
    denom = see + sff - 2.0 * sef
    if not np.isfinite(denom) or denom <= np.finfo(np.float64).eps:
        w_e = 0.5
    else:
        w_e = float(np.clip((sff - sef) / denom, 0.0, 1.0))
    w_f = 1.0 - w_e
    return dict(w_evolution=w_e, w_assay=w_f, bias_evolution=bias_e, bias_assay=bias_f, residual_covariance=[[see, sef], [sef, sff]], oof_rmse_evolution=float(np.sqrt(np.mean(err_e ** 2))), oof_rmse_assay=float(np.sqrt(np.mean(err_f ** 2))))

def fused_select(data, benchmark, train, y, keys, fit_seed, rng, algorithm_seed, rnd, evidence, weights):
    (e_full, e_meta) = expert_members(data, benchmark, 'evolution', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    (f_full, f_meta) = expert_members(data, benchmark, 'assay', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    e_full = e_full + weights['bias_evolution']
    f_full = f_full + weights['bias_assay']
    (we, wf) = (weights['w_evolution'], weights['w_assay'])
    (kind, head) = FINE_RECIPE[benchmark]
    if head == 'rf100':
        combined = we * e_full.mean(axis=1) + wf * f_full[:, 0]
        order = np.lexsort((keys, -combined))[:BATCH]
        choices = [(int(j), -1) for j in order]
        scores = combined[:, None]
        selector = 'greedy_same_as_rf100_fine'
    else:
        if e_full.shape[1] != 5 or f_full.shape[1] != 5:
            raise RuntimeError('TS fusion requires 5 members per expert')
        combined = we * e_full + wf * f_full
        choices = ts_select(keys, combined, BATCH, rng)
        scores = combined
        selector = 'TS96_same_as_frozen_fine'
    meta = dict(selection='minimum_variance_dual_expert', selector=selector, fine_representation=kind, fine_head=head, evolution_model=e_meta, assay_model=f_meta, **weights)
    return (choices, scores, meta)

def fine_only_select(data, benchmark, train, y, keys, fit_seed, rng, algorithm_seed, rnd):
    (kind, head) = FINE_RECIPE[benchmark]
    (scores, meta) = expert_members(data, benchmark, 'assay', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    if head == 'rf100':
        order = np.lexsort((keys, -scores[:, 0]))[:BATCH]
        choices = [(int(j), -1) for j in order]
        selector = 'greedy'
    else:
        choices = ts_select(keys, scores, BATCH, rng)
        selector = 'TS96'
    return (choices, scores, dict(selection='matched_fine_only', selector=selector, fine_representation=kind, fine_head=head, assay_model=meta))

def evolution_only_select(data, b, train, y, keys, fit_seed, rng, algorithm_seed, rnd):
    (ep, em) = expert_members(data, b, 'evolution', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    er = rank_members(ep)
    (kind, head) = FINE_RECIPE[b]
    if head == 'rf100':
        mean = er.mean(1)
        order = np.lexsort((keys, -mean))[:96]
        ch = [(int(j), -1) for j in order]
        selector = 'greedy'
    else:
        ch = ts_select(keys, er, 96, rng)
        selector = 'TS96'
    return (ch, keys, er, dict(screening='none', selection='evolution_only_rank', selector=selector, evolution_model=em, fine_candidate_n=len(keys)))
