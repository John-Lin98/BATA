"""Input-contract checks: no real assay data or scientific experiments."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from bata.__main__ import observed_rows


class ProtocolTests(unittest.TestCase):
    def test_four_round_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(7)
            for name in ('dca','task'):
                np.save(root/(name+'.npy'),rng.normal(size=(512,8)))
            observations={i:float(rng.normal()) for i in range(96)}
            history=[max(observations.values())]
            for rnd in range(1,5):
                observed=root/f'observed{rnd}.csv'
                with observed.open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['index','fitness'])
                    writer.writerows(sorted(observations.items()))
                output=root/f'batch{rnd}.json'
                subprocess.run([sys.executable,'-m','bata','--benchmark','GB1',
                    '--dca-features',str(root/'dca.npy'),'--task-features',str(root/'task.npy'),
                    '--observed',str(observed),'--algorithm-seed','31','--round',str(rnd),
                    '--output',str(output)],check=True,capture_output=True,text=True)
                batch=json.loads(output.read_text())['selected_indices']
                self.assertEqual(len(set(batch)),96)
                self.assertFalse(set(batch).intersection(observations))
                # Generate toy measurements only after the complete batch has been emitted.
                feedback={i:float(rng.normal()) for i in batch}
                observations.update(feedback);history.append(max(observations.values()))
            self.assertEqual(len(observations),480)
            self.assertEqual(history,sorted(history))

    def test_observed_only_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observed.csv'
            def write(rows, header=('index', 'fitness')):
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
            rows = [(i, i / 100) for i in range(96)]
            write(rows)
            ids, y = observed_rows(path, 224, 1)
            self.assertEqual(len(ids), 96)
            for bad in (rows + [(100, 1.)], rows[:-1], rows[:-1] + [rows[0]],
                        rows[:-1] + [(500, 1.)], rows[:-1] + [(95, float('nan'))]):
                write(bad)
                with self.assertRaises(ValueError): observed_rows(path, 224, 1)
            write(rows, ('index', 'hidden_fitness'))
            with self.assertRaises(ValueError): observed_rows(path, 224, 1)


if __name__ == '__main__':
    unittest.main()
