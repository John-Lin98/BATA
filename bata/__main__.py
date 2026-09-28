"""Select a batch from supplied features and already measured feedback only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .core import FINE_RECIPE, crossfit_oof, bata_weight, rank_fused_select
from .calibration import OBJECTIVES, objective_weight
from .variants import fine_choose, gate_choose, recipe_key
from .baselines import forest_select


def observed_rows(path, size, round_number):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['index', 'fitness']:
            raise ValueError('Observed CSV must contain only index,fitness')
        rows = [(int(row['index']), float(row['fitness'])) for row in reader]
    rows.sort()
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    values = np.asarray([r[1] for r in rows], dtype=np.float64)
    if len(ids) != 96 * round_number or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected exactly 96 * round distinct revealed observations')
    if np.any(ids < 0) or np.any(ids >= size) or not np.isfinite(values).all():
        raise ValueError('Invalid revealed index or fitness')
    return ids, values


class Features:
    def __init__(self, dca, task, kind, tokens=False):
        self.tokens = tokens
        if tokens and kind != 'onehot':
            raise ValueError('Token input is supported only for a one-hot task expert')
        self.arrays = {'dca': np.load(dca, mmap_mode='r', allow_pickle=False),
                       kind: np.load(task, mmap_mode='r', allow_pickle=False)}
        sizes = set()
        for array in self.arrays.values():
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
                raise ValueError('Features must be numeric two-dimensional NPY arrays')
            sizes.add(len(array))
        if len(sizes) != 1:
            raise ValueError('Feature arrays must share the same candidate row order and size')
        self.size = sizes.pop()

    def feature(self, kind, ids):
        array = np.asarray(self.arrays[kind][ids])
        if not np.isfinite(array).all():
            raise ValueError('Non-finite features')
        if kind == 'onehot' and self.tokens:
            if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(array >= 20):
                raise ValueError('Invalid amino-acid tokens')
            return np.eye(20, dtype=np.float64)[array].reshape(len(array), -1)
        return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=sorted(set(FINE_RECIPE) | {'GRB2'}), required=True)
    parser.add_argument('--method', choices=('bata','fine_m20','g20','g30','g40','rf_onehot','evolvepro650','alde','reap100',
                                             'matched_fine','evolution_only','equal_rank','global_mse'), default='bata')
    parser.add_argument('--objective', choices=OBJECTIVES, default='T-DCG-96',
                        help='Default is original BATA; alternatives are matched mechanism controls')
    parser.add_argument('--dca-features', type=Path, required=True)
    task_input = parser.add_mutually_exclusive_group(required=True)
    task_input.add_argument('--task-features', type=Path)
    task_input.add_argument('--tokens', type=Path, help='Integer NPY tokens; lazily form original float64 one-hot features')
    parser.add_argument('--observed', type=Path, required=True,
                        help='Only previously revealed index,fitness rows; never full landscape labels')
    parser.add_argument('--algorithm-seed', type=int, required=True)
    parser.add_argument('--round', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.method != 'bata' and args.objective != 'T-DCG-96':
        parser.error('Objective variants apply only to BATA, not expert-sufficiency controls')
    if args.output.exists():
        parser.error('Output exists; use a new path')
    benchmark_key = recipe_key(args.benchmark)
    task_kind = {'rf_onehot': 'onehot', 'evolvepro650': 'plm', 'alde': 'onehot', 'reap100': 'plm'}.get(
        args.method, FINE_RECIPE[benchmark_key][0])
    if args.tokens is not None and task_kind != 'onehot':
        parser.error('This method/benchmark requires PLM --task-features, not --tokens')
    data = Features(args.dca_features, args.tokens if args.tokens is not None else args.task_features,
                    task_kind, tokens=args.tokens is not None)
    train, y = observed_rows(args.observed, data.size, args.round)
    keys = np.setdiff1d(np.arange(data.size), train, assume_unique=True)
    if len(keys) < 96:
        parser.error('At least 96 unmeasured candidates are required')
    seed = int(np.random.SeedSequence([args.algorithm_seed, args.round, 1701]).generate_state(1)[0])
    rng = np.random.default_rng(np.random.SeedSequence([args.algorithm_seed, args.round, 73]))
    if args.method in ('alde', 'reap100'):
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(limits=1):
        if args.method == 'bata':
            prior, task, _ = crossfit_oof(data, benchmark_key, train, y, seed, args.algorithm_seed, args.round)
            weight = objective_weight(y, prior, task, args.objective)
            chosen, scores, _ = rank_fused_select(data, benchmark_key, train, y, keys,
                seed, rng, args.algorithm_seed, args.round, weight['w_evolution'], 'bata')
            prior_weight = weight['w_evolution']
        elif args.method in ('matched_fine', 'evolution_only', 'equal_rank', 'global_mse'):
            from . import controls
            common = (data, benchmark_key, train, y, keys, seed, rng, args.algorithm_seed, args.round)
            if args.method == 'matched_fine':
                chosen, scores, _ = controls.fine_only_select(*common)
                prior_weight = 0.0
            elif args.method == 'evolution_only':
                chosen, _, scores, _ = controls.evolution_only_select(*common)
                prior_weight = 1.0
            elif args.method == 'equal_rank':
                chosen, scores, _ = rank_fused_select(*common, 0.5, 'equal_rank_dual_expert')
                prior_weight = 0.5
            else:
                evidence = controls.crossfit_evidence(data, benchmark_key, train, y, keys, seed,
                                                      args.algorithm_seed, args.round, need_candidate_envelopes=False)
                weights = controls.min_variance_weights(y, evidence['oof_e'], evidence['oof_f'])
                chosen, scores, _ = controls.fused_select(*common, evidence, weights)
                prior_weight = weights['w_evolution']
        elif args.method == 'alde':
            from .alde import alde_select
            chosen, scores, _ = alde_select(data, train, y, keys, seed, rng)
            prior_weight = None
        elif args.method == 'reap100':
            from .reap_adapter import reap_select
            chosen, scores, _ = reap_select(data, train, y, keys, args.algorithm_seed, args.round)
            prior_weight = None
        elif args.method in ('rf_onehot', 'evolvepro650'):
            chosen, scores, _ = forest_select(data, train, y, keys, seed, args.method)
            prior_weight = None
        elif args.method == 'fine_m20':
            chosen, _, scores, _ = fine_choose(args.benchmark, data, train, y, keys,
                seed, rng, args.algorithm_seed, args.round)
            prior_weight = None
        else:
            chosen, _, scores, meta = gate_choose(args.benchmark, args.method, data, train, y, keys,
                seed, rng, args.algorithm_seed, args.round)
            prior_weight = meta['gate_w_evolution']
    indices = [int(keys[j]) for j, _ in chosen]
    assert len(set(indices)) == 96 and not set(indices).intersection(train)
    result = dict(benchmark=args.benchmark, method=args.method, objective=args.objective, round=args.round, algorithm_seed=args.algorithm_seed,
                  prior_weight=prior_weight, selected_indices=indices,
                  members=[int(m) for _, m in chosen],
                  selected_scores=[scores[j].tolist() for j, _ in chosen],
                  label_access='provided revealed observations only; no oracle access')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
