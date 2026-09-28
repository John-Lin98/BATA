"""Frozen region-specific DCA arithmetic; external py-mfdca is not vendored.

Input codes use -ACDEFGHIKLMNPQRSTVWY (gap=0), not the 20-letter task tokens.
No sequence labels are used. GPU weight computation is only run when requested.
"""
import time
import numpy as np
import torch

AA = '-ACDEFGHIKLMNPQRSTVWY'
Q = 21
THETA = .2
PSEUDO = .5

def _gpu_weights(enc: np.ndarray, device: torch.device, block: int):
    N = enc.shape[1]
    max_diffs = round(THETA * N)
    min_matches = N - max_diffs
    x = torch.from_numpy(enc.astype(np.int64)).to(device)
    one = torch.nn.functional.one_hot(x, num_classes=Q).reshape(len(enc), -1).to(torch.float16)
    counts = np.empty(len(enc), dtype=np.int64)
    torch.cuda.reset_peak_memory_stats(device)
    t = time.perf_counter()
    with torch.inference_mode():
        xt = one.T.contiguous()
        for start in range(0, len(enc), block):
            sim = one[start:start + block] @ xt
            c = (sim >= min_matches).sum(dim=1)
            counts[start:start + len(c)] = c.cpu().numpy()
    torch.cuda.synchronize(device)
    sec = time.perf_counter() - t
    if (counts < 1).any():
        raise RuntimeError('invalid redundancy count')
    k = min(32, len(enc))
    direct = ((enc[:k, None, :] == enc[None, :, :]).sum(axis=2) >= min_matches).sum(axis=1)
    if not np.array_equal(direct, counts[:k]):
        raise RuntimeError('GPU identity counts do not match exact CPU validation')
    weights = 1.0 / counts.astype(np.float64)
    return (weights, counts, dict(theta=THETA, max_diffs=max_diffs, min_matches=min_matches, seconds=sec, block=block, peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2 ** 20, cpu_validation_rows=k))

def _fit_dca(enc: np.ndarray, weights: np.ndarray):
    N = enc.shape[1]
    import numba
    numba.set_num_threads(min(8, numba.get_num_threads()))
    from dca.dca_functions import Compute_AverageLocalField, Compute_Results, add_pseudocount, computeC, compute_Pi, compute_Pij, invC_to_4D
    seq = enc.astype(np.int64, copy=False)
    m = len(seq)
    meff = float(weights.sum())
    t = time.perf_counter()
    pi = compute_Pi(seq, PSEUDO, N, m, Q, meff, weights)
    pij = compute_Pij(seq, PSEUDO, N, m, Q, meff, weights, pi)
    (pi_pc, pij_pc) = add_pseudocount(pi, pij, PSEUDO, N, Q)
    c = computeC(pi_pc, pij_pc, N, Q)
    if not np.isfinite(c).all():
        raise RuntimeError('nonfinite connected correlations')
    eig_min = float(np.linalg.eigvalsh(c).min())
    invc = np.linalg.inv(c)
    couplings = invC_to_4D(-invc, N, Q)
    (pairfields, di) = Compute_Results(pi_pc, -couplings, N, Q)
    localfields = Compute_AverageLocalField(pairfields, N, Q)
    sec = time.perf_counter() - t
    if couplings.shape != (N, N, Q, Q) or localfields.shape != (Q, N):
        raise RuntimeError('DCA parameter shape mismatch')
    if not np.isfinite(couplings).all() or not np.isfinite(localfields).all():
        raise RuntimeError('nonfinite DCA parameters')
    return (localfields, couplings, di, dict(seconds=sec, meff=meff, covariance_min_eigenvalue=eig_min, numba_threads=numba.get_num_threads()))

def _features(seq_idx: np.ndarray, localfields: np.ndarray, couplings: np.ndarray, batch=4096):
    N = seq_idx.shape[1]
    if seq_idx.ndim != 2 or seq_idx.shape[1] != N:
        raise ValueError('invalid sequence matrix')
    result = np.empty((len(seq_idx), N), dtype=np.float32)
    j = np.arange(N)[None, :]
    t = time.perf_counter()
    for start in range(0, len(seq_idx), batch):
        s = seq_idx[start:start + batch].astype(np.int64, copy=False)
        b = len(s)
        f = np.empty((b, N), dtype=np.float64)
        for i in range(N):
            ai = s[:, i]
            vals = couplings[i, j, ai[:, None], s]
            pair = vals.sum(axis=1) - couplings[i, i, ai, ai]
            f[:, i] = localfields[ai, i] + 0.5 * pair
        result[start:start + b] = f.astype(np.float32)
    sec = time.perf_counter() - t
    if not np.isfinite(result).all():
        raise RuntimeError('nonfinite DCA features')
    return (result, sec)
