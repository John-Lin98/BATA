"""Post-hoc appendix metrics; never import this evaluator in the optimizer."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bata.data import sha256


def endpoint_metrics(truth, world, batches):
    ids = [i for batch in batches for i in batch]
    if (len(batches) != 3 or any(len(b) != 16 for b in batches) or len(set(ids)) != 48
            or any(type(i) is not int for i in ids) or not set(ids).issubset(set(world))):
        raise ValueError('Expected three disjoint batches of 16 within the frozen world')
    world = np.asarray(world, dtype=np.int64)
    wtruth = np.asarray(truth, dtype=float)[world]
    if not np.isfinite(wtruth).all():
        raise ValueError('Nonfinite world labels')
    ranks = pd.Series(wtruth).rank(pct=True).to_numpy()
    local = {int(idx): k for k, idx in enumerate(world)}
    pct = np.array([ranks[local[i]] for i in ids], float)
    values = np.array([truth[i] for i in ids], float)
    lo, hi = float(wtruth.min()), float(wtruth.max())
    best = float(values.max())
    return dict(queries=48, top10=int(np.sum(pct >= .90)), hit1=bool(np.any(pct >= .99)),
                final=best, best_percentile=float(pct.max()),
                normalized_regret=(hi-best)/max(hi-lo, np.finfo(float).eps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assay-csv', type=Path, required=True)
    parser.add_argument('--batches', type=Path, nargs=3, required=True, help='Round1/2/3 JSON outputs in order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((ROOT / 'configs/DATA_IDENTITY.json').read_text())['PABP']
    if sha256(args.assay_csv) != identity['source_artifacts']['fitness_csv']['sha256']:
        raise ValueError('Frozen assay SHA mismatch')
    outputs = [json.loads(p.read_text()) for p in args.batches]
    sim, arm = outputs[0]['sim'], outputs[0]['arm']
    if any((o['sim'], o['arm'], o['round'], o['protocol']) != (sim, arm, r, '3x16')
           for r, o in enumerate(outputs, 1)):
        raise ValueError('Mixed or unordered batch outputs')
    manifest = json.loads((ROOT / 'manifests/pabp_low_budget48.json').read_text())
    entry = next(e for e in manifest['entries'] if (e['sim'], e['arm']) == (sim, arm))
    if outputs[0]['selected_indices'] != entry['initial_indices']:
        raise ValueError('Frozen initial batch mismatch')
    truth = pd.read_csv(args.assay_csv, usecols=['DMS_score'])['DMS_score'].to_numpy(float)
    result = endpoint_metrics(truth, manifest['worlds'][str(sim)], [o['selected_indices'] for o in outputs])
    result.update(sim=sim, arm=arm, evaluation='post-hoc only; full-world truth not available to selector')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
