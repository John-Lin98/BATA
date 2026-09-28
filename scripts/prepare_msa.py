"""Reconstruct frozen MSA inputs without fitting or labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bata.data import sha256, array_sha256
from bata.msa import region_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['GB1', 'TrpB', 'PABP', 'HIS7', 'GRB2'], required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'configs/MSA_RECIPE.json'
    recipe = json.loads(manifest.read_text())[args.benchmark]
    if sha256(args.alignment) != recipe['source_sha256']:
        raise ValueError('Frozen alignment SHA mismatch')
    codes, raw_count = region_codes(args.alignment, recipe['full_length'],
                                   recipe['start'], recipe['stop'], recipe.get('mode', 'aligned'))
    if (raw_count != recipe['raw_count'] or list(codes.shape) != recipe['shape']
            or array_sha256(codes) != recipe['encoded_sha256']):
        raise ValueError('Frozen MSA reconstruction mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'region_msa.npy', codes, allow_pickle=False)
    receipt = dict(recipe, benchmark=args.benchmark, exact_array_parity=True)
    (args.output / 'MSA_IDENTITY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'benchmark': args.benchmark, 'shape': list(codes.shape), 'exact_array_parity': True}))


if __name__ == '__main__':
    main()
