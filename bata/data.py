"""Frozen feature identity checks; never opens a fitness/oracle file."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), 8192):
        digest.update(np.ascontiguousarray(array[start:start+8192]).tobytes())
    return digest.hexdigest()


def verified_features(identity, domain, tokens, dca, plm):
    for key, path in [('public_domain', domain), ('tokens', tokens), ('dca_features', dca)]:
        if sha256(path) != identity['source_artifacts'][key]['sha256']:
            raise ValueError('Frozen source SHA mismatch: ' + key)
    if sha256(plm) != identity['plm_sha256']:
        raise ValueError('Frozen PLM SHA mismatch')
    combos = json.loads(Path(domain).read_text())['combos']
    ordered = hashlib.sha256(json.dumps(combos, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    count = identity['candidate_count']
    if len(combos) != count or len(set(combos)) != count or ordered != identity['ordered_candidate_identity_sha256']:
        raise ValueError('Candidate order/count mismatch')
    arrays = {}
    for key, path, field, shape in [('tokens', tokens, 'tokens', identity['tokens_shape']),
                                    ('dca', dca, 'features', identity['dca_shape'])]:
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive['indices'], np.arange(count)):
                raise ValueError('Feature row indices mismatch: ' + key)
            array = archive[field]
            if list(array.shape) != shape or not np.isfinite(array).all():
                raise ValueError('Feature shape/values mismatch: ' + key)
            arrays[key] = array
    if (not np.issubdtype(arrays['tokens'].dtype, np.integer)
            or np.any(arrays['tokens'] < 0) or np.any(arrays['tokens'] >= 20)):
        raise ValueError('Expected canonical 20-amino-acid integer tokens')
    embedding = np.load(plm, mmap_mode='r', allow_pickle=False)
    if list(embedding.shape) != identity['plm_shape']:
        raise ValueError('PLM shape mismatch')
    return arrays
