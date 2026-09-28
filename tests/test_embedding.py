import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch not installed')
class EmbeddingInput(unittest.TestCase):
    def test_substitution_and_pooling(self):
        import torch
        from bata.embedding import AA, validate_context, encode_rows
        class Alphabet:
            prepend_bos = True
            def get_idx(self, aa):return AA.index(aa)+2
            def get_batch_converter(self):
                return lambda rows:(None,None,torch.tensor([[0]+[self.get_idx(c) for c in s]+[1] for _,s in rows]))
        seen=[]
        def model(x, **kwargs):
            seen.append(x.clone())
            self.assertEqual(kwargs,dict(repr_layers=[33],return_contacts=False))
            return {'representations':{33:x.float().unsqueeze(-1).repeat(1,1,2)}}
        tokens=np.array([[19,0],[1,2]])
        validate_context('ACDE',[2,4],tokens)
        result=encode_rows(model,Alphabet(),'ACDE',[2,4],tokens,np.array([1,0]),'cpu')
        expected=np.array([[2,3,4,4],[2,21,4,2]])
        np.testing.assert_array_equal(seen[0].numpy()[:,1:-1],expected)
        np.testing.assert_array_equal(result,np.repeat(expected.mean(1)[:,None],2,axis=1))
        with self.assertRaises(ValueError):validate_context('ACDE',[2,2],tokens)
        with self.assertRaises(ValueError):validate_context('ACDE',[2,4],tokens+20)


if __name__ == '__main__':unittest.main()
