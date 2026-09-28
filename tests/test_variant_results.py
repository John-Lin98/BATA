"""The release check must reject a changed value or duplicated initialization."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_results.py'
SPEC = importlib.util.spec_from_file_location('release_verify', SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VariantResultsTest(unittest.TestCase):
    def test_calibration_summaries_and_pairing(self):
        self.assertEqual(VERIFY.verify_calibration(), (12, 18))
        original = VERIFY.rows
        def changed(path):
            data = original(path)
            if path.name == 'matched_objective_per_run.csv':
                data[0]['algorithm_seed'] = '0'
            return data
        with patch.object(VERIFY, 'rows', changed):
            with self.assertRaises(AssertionError):
                VERIFY.verify_calibration()

    def test_frozen_summaries(self):
        self.assertEqual(VERIFY.verify_variants(), 45)

    def test_engineering_speed_and_incomplete_pilot(self):
        self.assertEqual(VERIFY.verify_engineering_speed(), 15)
        original = VERIFY.rows
        def changed(path):
            data = original(path)
            if path.name == 'speed_comparison.csv':
                next(r for r in data if r['variant'] == 'fast_m5')['Final'] = '0'
            return data
        with patch.object(VERIFY, 'rows', changed):
            with self.assertRaises(AssertionError):
                VERIFY.verify_engineering_speed()

    def test_changed_result_and_duplicate_group_are_rejected(self):
        original = VERIFY.rows
        for corruption in ('value', 'duplicate'):
            def changed(path):
                data = original(path)
                if path.name == 'gate_development5_per_run.csv':
                    if corruption == 'value':
                        data[0]['Final'] = str(float(data[0]['Final']) + 0.01)
                    else:
                        data[1]['group'] = data[0]['group']
                return data
            with self.subTest(corruption=corruption), patch.object(VERIFY, 'rows', changed):
                with self.assertRaises(AssertionError):
                    VERIFY.verify_variants()
