import unittest
from bata.calibration import unit_test


class CalibrationTests(unittest.TestCase):
    def test_frozen_objective_checks(self):
        result=unit_test()
        self.assertTrue(result['valid'])
        self.assertEqual(result['synthetic_cases'],32)


if __name__=='__main__':
    unittest.main()
