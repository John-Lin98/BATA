import tempfile
import unittest
from pathlib import Path
import numpy as np
from bata.msa import region_codes


class TestMSA(unittest.TestCase):
    def test_frozen_cleaning_and_multiplicity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.a2m'
            path.write_text('>a\nAa.c\n>b\nAA-C\n>badgap\nA---\n>badletter\nAXXC\n')
            codes, count = region_codes(path, 4, 2, 4)
            self.assertEqual(count, 4)
            np.testing.assert_array_equal(codes, [[1, 0, 2], [1, 0, 2]])
            path.write_text('>wronglength\nAC\n')
            with self.assertRaises(ValueError):
                region_codes(path, 4, 2, 4)

    def test_jackhmmer_insertions_and_grouped_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.a2m'
            path.write_text('>a\nAxyC.D\n>b\nACE\n>c\nACD\n')
            codes, count = region_codes(path, 3, 1, 3, 'jackhmmer_grouped')
            self.assertEqual(count, 3)
            np.testing.assert_array_equal(codes, [[1, 2, 3], [1, 2, 3], [1, 2, 4]])


if __name__ == '__main__':
    unittest.main()
