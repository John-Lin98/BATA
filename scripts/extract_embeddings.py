"""Rebuild a full-context ESM2-650M feature shard; no labels or automatic downloads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bata.data import sha256, array_sha256
from bata.embedding import validate_context, encode_rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--context', type=Path, required=True, help='JSON with sequence and positions_1based only')
parser.add_argument('--tokens', type=Path, required=True, help='Ordered integer mutation tokens NPY')
parser.add_argument('--checkpoint', type=Path, required=True, help='Original externally obtained checkpoint; pinned SHA required')
parser.add_argument('--numeric-mode', choices=['fp32','3xbf16'], required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():parser.error('Output directory must be new')
context = json.loads(args.context.read_text())
if set(context) != {'sequence','positions_1based'}:
    parser.error('Context must contain only sequence and positions_1based')
tokens = np.load(args.tokens, mmap_mode='r', allow_pickle=False)
validate_context(context['sequence'],context['positions_1based'],tokens)
end = len(tokens) if args.end is None else args.end
if not 0 <= args.start < end <= len(tokens) or args.batch < 1:
    parser.error('Invalid row range or batch size')
checkpoint_sha = sha256(args.checkpoint)
if checkpoint_sha != 'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c':
    parser.error('Checkpoint differs from the frozen ESM2-650M artifact')
if torch.cuda.device_count() != 1:parser.error('Exactly one visible CUDA GPU required')
import esm
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Load only after checking the original externally acquired file identity.
state = torch.load(args.checkpoint,map_location='cpu')
model,alphabet = esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',state,None)
model = model.cuda().float().eval().requires_grad_(False)
def encode(ids):
    return encode_rows(model,alphabet,context['sequence'],context['positions_1based'],tokens,ids,'cuda')
controls = np.linspace(args.start,end-1,min(8,end-args.start),dtype=int)
error = None
if args.numeric_mode == '3xbf16':
    reference = encode(controls)
    from bata.embedding_kernel import install
    install(model)
    error = float(np.max(np.abs(encode(controls)-reference)))
    if error > 1e-4:raise RuntimeError('Accelerated feature qualification failed')
args.output.mkdir(parents=True,exist_ok=False)
features = np.lib.format.open_memmap(args.output/'embeddings.npy',mode='w+',dtype=np.float32,shape=(end-args.start,1280))
for off in range(args.start,end,args.batch):
    stop = min(end,off+args.batch)
    features[off-args.start:stop-args.start] = encode(np.arange(off,stop))
features.flush()
receipt = dict(labels_read=False,start=args.start,end=end,total_candidates=len(tokens),batch=args.batch,
    checkpoint_sha256=checkpoint_sha,context_sha256=sha256(args.context),tokens_sha256=sha256(args.tokens),
    numeric_mode=args.numeric_mode,control_maxabs=error,control_indices=controls.tolist(),
    pooling='Layer33 mean of all non-special full-sequence residues; float32 output',
    features_sha256=sha256(args.output/'embeddings.npy'),
    slice_sha256=array_sha256(features),
    qualification='Numerically qualified reconstruction, not assumed frozen-cache exact parity')
with (args.output/'EMBEDDING_REBUILD.json').open('x') as f:json.dump(receipt,f,indent=2)
print('Completed label-free feature shard; compare frozen identity before scientific replay')
