"""Rebuild label-blind DCA features from ordered region-code arrays.

Does not download data or expose assay labels. This is a user-invoked rebuild,
not a new benchmark experiment; no feature rebuild is run during packaging.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bata.data import sha256
from bata.dca_features import _gpu_weights, _fit_dca, _features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--region-msa', type=Path, required=True, help='Ordered MSA codes in NPY, alphabet -ACDEFGHIKLMNPQRSTVWY')
parser.add_argument('--candidate-regions', type=Path, required=True, help='Same alphabet, frozen candidate row order, NPY')
parser.add_argument('--weight-block', type=int, default=4096)
parser.add_argument('--feature-batch', type=int, default=4096)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or args.weight_block < 1 or args.feature_batch < 1:
    parser.error('Use a new output directory and positive block sizes')
msa = np.load(args.region_msa, allow_pickle=False)
candidates = np.load(args.candidate_regions, allow_pickle=False)
for name, value in [('MSA', msa), ('candidates', candidates)]:
    if (value.ndim != 2 or not value.size or not np.issubdtype(value.dtype, np.integer)
            or value.min() < 0 or value.max() > 20):
        parser.error(name + ' must be a nonempty integer matrix with codes 0..20')
if candidates.shape[1] != msa.shape[1] or np.any(candidates == 0):
    parser.error('Candidate regions must have the MSA width and no gaps')
if torch.cuda.device_count() != 1:
    parser.error('Exactly one visible GPU is required by the frozen weight recipe')
import dca.dca_functions as upstream
if sha256(upstream.__file__) != 'd6b518bacf77ba8aedf5d6f30822c8128aaa726f55f597365a8bb370a71ffcb1':
    parser.error('Installed py-mfdca arithmetic differs from the frozen source')
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
weights, _, weight_info = _gpu_weights(msa, torch.device('cuda'), args.weight_block)
fields, couplings, di, fit_info = _fit_dca(msa, weights)
features, seconds = _features(candidates, fields, couplings, args.feature_batch)
args.output.mkdir(parents=True, exist_ok=False)
np.save(args.output/'dca.npy', features, allow_pickle=False)
np.savez_compressed(args.output/'dca_model.npz', localfields=fields, couplings=couplings, DI=di)
receipt = dict(labels_read=False, candidate_count=len(candidates), region_length=msa.shape[1],
    msa_sha256=sha256(args.region_msa), candidate_regions_sha256=sha256(args.candidate_regions),
    features_sha256=sha256(args.output/'dca.npy'), weight_info=weight_info, fit_info=fit_info,
    feature_seconds=seconds, upstream_source_sha256=sha256(upstream.__file__),
    qualification='Rebuilt input; exact parity with the frozen cache must be checked separately')
with (args.output/'DCA_REBUILD.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print('Completed label-blind DCA rebuild; not a benchmark result')
