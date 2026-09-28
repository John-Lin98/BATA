import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bata.data import sha256, array_sha256, verified_features
from bata.__main__ import Features


class DataIdentity(unittest.TestCase):
    def test_identity_conversion_and_rejection(self):
        root = Path(tempfile.mkdtemp(prefix='bata-data-smoke-'))
        tokens = np.arange(24).reshape(12, 2) % 20
        self.assertEqual(array_sha256(tokens),hashlib.sha256(tokens.tobytes()).hexdigest())
        dca = np.arange(36.).reshape(12, 3)
        combos = ['candidate'+str(i) for i in range(12)]
        (root/'domain.json').write_text(json.dumps(dict(combos=combos)))
        np.savez(root/'tokens.npz', tokens=tokens, indices=np.arange(12))
        np.savez(root/'dca.npz', features=dca, indices=np.arange(12))
        np.save(root/'plm.npy', np.ones((12, 5), dtype=np.float32))
        identity = dict(candidate_count=12, tokens_shape=[12, 2], dca_shape=[12, 3], plm_shape=[12, 5],
            ordered_candidate_identity_sha256=hashlib.sha256(json.dumps(combos,separators=(',',':')).encode()).hexdigest(),
            plm_sha256=sha256(root/'plm.npy'),source_artifacts={key:dict(sha256=sha256(root/file)) for key,file in
                [('tokens','tokens.npz'),('dca_features','dca.npz'),('public_domain','domain.json')]})
        args = (root/'domain.json',root/'tokens.npz',root/'dca.npz',root/'plm.npy')
        output = verified_features(identity,*args)
        np.testing.assert_array_equal(output['tokens'], tokens)
        for key, array in output.items():np.save(root/(key+'.npy'),array)
        feature = Features(root/'dca.npy',root/'tokens.npy','onehot',tokens=True)
        np.testing.assert_array_equal(feature.feature('onehot',[2,5]),np.eye(20)[tokens[[2,5]]].reshape(2,-1))
        bad = dict(identity,plm_sha256='0'*64)
        with self.assertRaises(ValueError):verified_features(bad,*args)
        bad = dict(identity,ordered_candidate_identity_sha256='0'*64)
        with self.assertRaises(ValueError):verified_features(bad,*args)
        with self.assertRaises(ValueError):Features(root/'dca.npy',root/'tokens.npy','plm',tokens=True)


if __name__ == '__main__':unittest.main()
