"""Reconstruct frozen candidate identities and sequence codes, excluding label columns."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bata.data import sha256, array_sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark',required=True)
parser.add_argument('--assay-csv',type=Path,required=True)
parser.add_argument('--reference-fasta',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
config=Path(__file__).resolve().parents[1]/'configs'
identities=json.loads((config/'DATA_IDENTITY.json').read_text())
recipes=json.loads((config/'PLM_RECIPE.json').read_text())
if args.benchmark not in identities:parser.error('Unknown benchmark')
if args.output.exists():parser.error('Output directory must be new')
identity,recipe=identities[args.benchmark],recipes[args.benchmark]
if sha256(args.assay_csv)!=identity['source_artifacts']['fitness_csv']['sha256']:
    parser.error('Assay source differs from frozen input; no filtering/reordering is allowed')
lines=args.reference_fasta.read_text().splitlines()
if sum(line.startswith('>') for line in lines)!=1:parser.error('Exactly one reference FASTA record required')
sequence=''.join(line.strip() for line in lines if not line.startswith('>')).upper()
if hashlib.sha256(sequence.encode()).hexdigest()!=recipe['reference_sequence_sha256']:
    parser.error('Reference sequence differs from frozen context')
aa='ACDEFGHIKLMNPQRSTVWY';positions=np.array(recipe['positions_1based'])-1
if args.benchmark in ('GB1','TrpB'):
    column='AAs' if args.benchmark=='GB1' else 'Combo'
    ids=pd.read_csv(args.assay_csv,usecols=[column])[column].astype(str).tolist()
    token_strings=ids
else:
    frame=pd.read_csv(args.assay_csv,usecols=['mutant','mutated_sequence'])
    ids=frame['mutant'].astype(str).tolist()
    outside=np.ones(len(sequence),bool);outside[positions]=False
    reference_outside=''.join(c for c,keep in zip(sequence,outside) if keep)
    token_strings=[]
    for seq in frame['mutated_sequence'].astype(str):
        if len(seq)!=len(sequence) or ''.join(c for c,keep in zip(seq,outside) if keep)!=reference_outside:
            parser.error('Unexpected sequence change outside frozen design region')
        token_strings.append(''.join(seq[p] for p in positions))
ordered=hashlib.sha256(json.dumps(ids,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
if (len(ids)!=identity['candidate_count'] or len(set(ids))!=len(ids)
        or ordered!=identity['ordered_candidate_identity_sha256']):
    parser.error('Candidate domain/order differs from frozen identity')
if any(len(s)!=len(positions) or any(c not in aa for c in s) for s in token_strings):
    parser.error('Noncanonical or wrong-length mutation tokens')
lut={c:i for i,c in enumerate(aa)}
tokens=np.array([[lut[c] for c in s] for s in token_strings],dtype=np.uint8)
lo,hi=int(positions.min()),int(positions.max())
region=np.broadcast_to(np.array([lut[c]+1 for c in sequence[lo:hi+1]],dtype=np.uint8),(len(ids),hi-lo+1)).copy()
region[:,positions-lo]=tokens+1
args.output.mkdir(parents=True,exist_ok=False)
np.save(args.output/'tokens.npy',tokens,allow_pickle=False)
np.save(args.output/'candidate_regions.npy',region,allow_pickle=False)
with (args.output/'context.json').open('x') as f:json.dump(dict(sequence=sequence,positions_1based=recipe['positions_1based']),f)
with (args.output/'candidate_ids.json').open('x') as f:json.dump(ids,f)
with (args.output/'SEQUENCE_IDENTITY.json').open('x') as f:
    json.dump(dict(benchmark=args.benchmark,candidate_count=len(ids),ordered_candidate_identity_sha256=ordered,
        reference_sequence_sha256=recipe['reference_sequence_sha256'],tokens_array_sha256=array_sha256(tokens),
        source_sha256=sha256(args.assay_csv),label_columns_parsed=False),f,indent=2)
print('PASS: frozen candidate identity/order reconstructed without parsing label columns')
