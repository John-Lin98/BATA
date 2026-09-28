"""Paper appendix PABP 3x16 adapter, not the main 480-query protocol."""
import numpy as np
from scipy.stats import rankdata
from .core import percentile_rank, crossfit_oof, expert_members, align_members, rank_members
BATCH = 16


def bata_weight16(y, oe, of):
    y = np.asarray(y, float)
    target = percentile_rank(y)
    re_ = percentile_rank(oe)
    rf = percentile_rank(of)
    k = min(BATCH, len(y))
    pos = rankdata(-y, method='average')
    w = np.where(pos <= k, 1.0 / np.log2(pos + 1.0), 0.0)
    d = re_ - rf
    t = target - rf
    den = float(np.sum(w * d * d))
    num = float(np.sum(w * d * t))
    we = 0.0 if den <= np.finfo(float).eps else float(np.clip(num / den, 0, 1))
    return dict(w_evolution=we, w_assay=1 - we, k_tail=k)

def bata_select16(data, train, y, keys, fit_seed, algorithm_seed, rnd):
    (oe, of, folds) = crossfit_oof(data, 'PABP', train, y, fit_seed, algorithm_seed, rnd)
    w = bata_weight16(y, oe, of)
    (ep, em) = expert_members(data, 'PABP', 'evolution', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    (fp, fm) = expert_members(data, 'PABP', 'assay', train, y, keys, fit_seed, algorithm_seed, rnd, fold_tag=None)
    (er, fr) = align_members(rank_members(ep), rank_members(fp))
    combined = w['w_evolution'] * er + w['w_assay'] * fr
    mean = combined.mean(1)
    order = np.lexsort((keys, -mean))[:BATCH]
    return ([(int(j), -1) for j in order], combined, dict(selection='BATA-FolDE-16', tail_arbitration=w, crossfit_folds=5, folds=folds, evolution_model=em, assay_model=fm))
