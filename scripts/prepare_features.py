"""Verify frozen inputs and adapt NPZ arrays for the label-free CLI.

This is a cache-format bridge, not upstream dataset acquisition or PLM extraction.
No labels, datasets, embeddings or model weights are downloaded or redistributed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bata.data import verified_features, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark', required=True)
for name in ('domain', 'tokens', 'dca', 'plm', 'output'):
    parser.add_argument('--'+name, type=Path, required=True)
args = parser.parse_args()
identities = json.loads((Path(__file__).resolve().parents[1]/'configs/DATA_IDENTITY.json').read_text())
if args.benchmark not in identities:
    parser.error('Unknown benchmark')
if args.output.exists():
    parser.error('Output must be a new directory')
identity = identities[args.benchmark]
arrays = verified_features(identity, args.domain, args.tokens, args.dca, args.plm)
args.output.mkdir(parents=True, exist_ok=False)
for name, array in arrays.items():
    with (args.output/(name+'.npy')).open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
receipt = dict(benchmark=args.benchmark, candidate_count=identity['candidate_count'],
    ordered_candidate_identity_sha256=identity['ordered_candidate_identity_sha256'],
    outputs={name+'.npy':sha256(args.output/(name+'.npy')) for name in arrays},
    plm_sha256=identity['plm_sha256'], labels_read=False,
    note='PLM array remains external at its original path; no copy made')
with (args.output/'FEATURE_IDENTITY.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print('PASS: frozen feature sources and row order verified; labels not accessed')
