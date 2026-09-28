"""Rebuild one prespecified GB1 reference round; require exact released-state hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bata import core
from bata.__main__ import Features, observed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dca-features', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True,
                        help='Current round revealed index,fitness CSV from frozen replay')
    parser.add_argument('--round', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--reference-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory')
    expected = json.loads(args.reference_receipt.read_text())
    if (expected.get('selection') != 'main35 GB1 group0, prespecified'
            or expected.get('passed') is not True):
        raise ValueError('Receipt is not the validated prespecified reference')
    kernel_sha = hashlib.sha256((ROOT/'bata/core.py').read_bytes()).hexdigest()
    if kernel_sha != expected['public_core_sha256']:
        raise ValueError('Kernel differs from the validated reference')
    entries = json.loads((ROOT/'manifests/main35.json').read_text())['entries']
    selected = [r for r in entries if r['benchmark'] == 'GB1' and r['group'] == 0 and r['method'] == 'BATA']
    if len(selected) != 1:
        raise ValueError('Expected exactly one fixed reference manifest')
    entry = selected[0]
    if entry['source_manifest_sha256'] != expected['source_manifest_sha256']:
        raise ValueError('Initialization manifest differs from the reference')
    data = Features(args.dca_features, args.tokens, 'onehot', tokens=True)
    if data.size != 149361:
        raise ValueError('Reference requires the frozen GB1 candidate domain')
    train, y = observed_rows(args.observed, data.size, args.round)
    if not set(entry['initial_indices']).issubset(train):
        raise ValueError('Revealed observations do not contain the fixed initialization')
    algorithm_seed = entry['algorithm_seed']
    seed = int(np.random.SeedSequence([algorithm_seed, args.round, 1701]).generate_state(1)[0])
    payload = {}
    with threadpool_limits(limits=1):
        prior, task, _ = core.crossfit_oof(data, 'GB1', train, y, seed, algorithm_seed, args.round)
        weight = core.bata_weight(y, prior, task)['w_evolution']
        checks = [r for r in expected['rounds'] if r['round'] == args.round]
        if len(checks) != 1 or weight != checks[0]['weight']:
            raise ValueError('Round calibration does not match the frozen reference')
        for name, kind in (('prior', 'dca'), ('task', 'onehot')):
            state = core.fit_bootstrap_ridge(data.feature(kind, train), y, seed, kind)
            buffer = io.BytesIO()
            np.savez(buffer, **state)
            content = buffer.getvalue()
            filename = f'round{args.round}_{name}.npz'
            digest = hashlib.sha256(content).hexdigest()
            if digest != expected['state_sha256'][filename]:
                raise ValueError(f'Rebuilt state differs from frozen reference: {filename}')
            with np.load(io.BytesIO(content), allow_pickle=False) as restored:
                for field in state:
                    np.testing.assert_array_equal(state[field], restored[field])
            payload[filename] = content
    # No output is written until both states and the calibration match exactly.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in payload.items():
        with (args.output/name).open('xb') as handle:
            handle.write(content)
    receipt = dict(benchmark='GB1', cohort='main35', group=0, round=args.round,
                   method='BATA original M5', prior_weight=weight,
                   public_core_sha256=kernel_sha, new_scientific_queries=0,
                   reference_receipt_sha256=hashlib.sha256(args.reference_receipt.read_bytes()).hexdigest(),
                   state_sha256={k: hashlib.sha256(v).hexdigest() for k,v in payload.items()},
                   exact_reference_state_bytes=True, exact_round_calibration=True,
                   scope='state rebuild only; frozen selection parity is recorded in the reference receipt')
    with (args.output/'REBUILD_RECEIPT.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(f'PASS: round {args.round}, two exact reference state files')


if __name__ == '__main__':
    main()
