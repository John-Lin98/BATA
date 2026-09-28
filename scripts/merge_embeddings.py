"""Verify reconstructed shards against frozen identities and merge; no GPU work."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bata.data import sha256, array_sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark',required=True)
parser.add_argument('--shards',type=Path,nargs='+',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
recipes=json.loads((Path(__file__).resolve().parents[1]/'configs/PLM_RECIPE.json').read_text())
if args.benchmark not in recipes:parser.error('Unknown benchmark')
if args.output.exists():parser.error('Output directory must be new')
recipe=recipes[args.benchmark]
shards=[]
for folder in args.shards:
    receipt=json.loads((folder/'EMBEDDING_REBUILD.json').read_text())
    shards.append((receipt,folder))
shards.sort(key=lambda pair:pair[0]['start'])
if len(shards)!=len(recipe['partitions']):parser.error('Wrong number of frozen partitions')
contexts=set();tokens=set()
for (receipt,folder),expected in zip(shards,recipe['partitions']):
    for key in ('start','end','batch','slice_sha256'):
        if receipt[key]!=expected[key]:parser.error('Rebuilt shard differs from frozen '+key)
    if (receipt['checkpoint_sha256']!=recipe['checkpoint_sha256']
            or receipt['numeric_mode']!=recipe['numeric_mode']
            or receipt['total_candidates']!=recipe['candidate_count']
            or receipt['labels_read'] is not False):parser.error('Incompatible shard protocol')
    array=np.load(folder/'embeddings.npy',mmap_mode='r',allow_pickle=False)
    if array.shape!=(expected['end']-expected['start'],1280) or array.dtype!=np.float32:
        parser.error('Invalid shard shape/dtype')
    if sha256(folder/'embeddings.npy')!=receipt['features_sha256'] or array_sha256(array)!=expected['slice_sha256']:
        parser.error('Shard bytes do not match receipt/frozen array identity')
    contexts.add(receipt['context_sha256']);tokens.add(receipt['tokens_sha256'])
if len(contexts)!=1 or len(tokens)!=1:parser.error('Mixed reference contexts or token arrays')
args.output.mkdir(parents=True,exist_ok=False)
merged=np.lib.format.open_memmap(args.output/'embeddings.npy',mode='w+',dtype=np.float32,shape=(recipe['candidate_count'],1280))
for receipt,folder in shards:
    array=np.load(folder/'embeddings.npy',mmap_mode='r',allow_pickle=False)
    merged[receipt['start']:receipt['end']]=array
merged.flush()
actual=sha256(args.output/'embeddings.npy')
if actual!=recipe['features_sha256']:
    raise RuntimeError('Merged file differs from frozen SHA; retained for diagnosis, not accepted')
with (args.output/'FEATURE_PARITY.json').open('x') as f:
    json.dump(dict(benchmark=args.benchmark,features_sha256=actual,all_frozen_slices_exact=True,labels_read=False),f,indent=2)
print('PASS: all frozen shard and merged feature identities match')
