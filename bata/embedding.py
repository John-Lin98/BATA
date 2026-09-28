"""Full-context mutation token construction without assay-label access."""
import numpy as np
import torch

AA = 'ACDEFGHIKLMNPQRSTVWY'


def validate_context(sequence, positions, tokens):
    if not sequence or any(c not in AA for c in sequence):
        raise ValueError('Reference must contain canonical amino acids only')
    if (not positions or any(type(p) is not int for p in positions)
            or len(set(positions)) != len(positions) or min(positions) < 1
            or max(positions) > len(sequence)):
        raise ValueError('Positions must be distinct one-based reference positions')
    if (tokens.ndim != 2 or tokens.shape[1] != len(positions) or not len(tokens)
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0) or np.any(tokens >= 20)):
        raise ValueError('Mutation token matrix must contain integer codes 0..19')


def encode_rows(model, alphabet, sequence, positions, tokens, ids, device):
    _, _, base = alphabet.get_batch_converter()([('ref', sequence)])
    base = base.to(device)
    amap = torch.tensor([alphabet.get_idx(a) for a in AA], device=device)
    sites = torch.tensor(np.asarray(positions)-1+int(alphabet.prepend_bos), device=device)
    x = base.repeat(len(ids), 1)
    x[:, sites] = amap[torch.as_tensor(tokens[ids].astype(np.int64), device=device)]
    with torch.inference_mode():
        result = model(x, repr_layers=[33], return_contacts=False)['representations'][33]
        pooled = result[:, 1:1+len(sequence)].float().mean(1)
    if not torch.isfinite(pooled).all():
        raise ValueError('Non-finite PLM embeddings')
    return pooled.cpu().numpy()
