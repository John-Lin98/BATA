"""One frozen PABP low-budget batch; no full-landscape label input."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bata.__main__ import Features
from bata.low_budget import bata_select16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim', type=int, choices=range(10), required=True)
    parser.add_argument('--arm', choices=['random_init', 'zero_shot_init'], required=True)
    parser.add_argument('--round', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--observed', type=Path)
    parser.add_argument('--dca-features', type=Path)
    parser.add_argument('--task-features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    manifest = json.loads((ROOT / 'manifests/pabp_low_budget48.json').read_text())
    entry = next(e for e in manifest['entries'] if e['sim'] == args.sim and e['arm'] == args.arm)
    world = np.asarray(manifest['worlds'][str(args.sim)], dtype=np.int64)
    if args.round == 1:
        if args.observed is not None:
            parser.error('Round 1 must not receive labels')
        indices, weight = entry['initial_indices'], None
    else:
        if any(p is None for p in (args.observed, args.dca_features, args.task_features)):
            parser.error('Rounds 2/3 require revealed feedback and both feature arrays')
        with args.observed.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['index', 'fitness']:
                raise ValueError('Observed CSV requires index,fitness only')
            rows = sorted((int(r['index']), float(r['fitness'])) for r in reader)
        train = np.asarray([r[0] for r in rows], dtype=np.int64)
        y = np.asarray([r[1] for r in rows], dtype=float)
        if (len(train) != 16 * (args.round - 1) or len(set(train)) != len(train)
                or not set(train).issubset(world) or not set(entry['initial_indices']).issubset(train)
                or not np.isfinite(y).all()):
            raise ValueError('Invalid revealed prefix')
        data = Features(args.dca_features, args.task_features, 'plm')
        if data.size != 37708:
            raise ValueError('Expected frozen complete PABP feature domain')
        measured = set(train)
        keys = np.asarray([i for i in world if i not in measured], dtype=np.int64)
        seed = int(np.random.SeedSequence([entry['algorithm_seed'], args.round, 1701]).generate_state(1)[0])
        with threadpool_limits(limits=1):
            chosen, _, meta = bata_select16(data, train, y, keys, seed, entry['algorithm_seed'], args.round)
        indices = [int(keys[j]) for j, _ in chosen]
        weight = meta['tail_arbitration']['w_evolution']
    if len(indices) != 16 or len(set(indices)) != 16:
        raise ValueError('Invalid batch')
    with args.output.open('x') as stream:
        json.dump(dict(sim=args.sim, arm=args.arm, round=args.round,
                       selected_indices=indices, prior_weight=weight, protocol='3x16'), stream, indent=2)


if __name__ == '__main__':
    main()
