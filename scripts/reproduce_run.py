"""Replay one frozen main/sensitivity/calibration run; never sweeps or selects seeds."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bata.data import sha256
from bata.evaluation import closed_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['main35', 'sensitivity70', 'matched_objective24', 'ablation24', 'transfer24',
                                            'fine_m20_formal118', 'fine_m20_trpb70', 'g20_confirmation10', 'gate_development5'], required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--method', required=True, help='Exact method label in frozen manifest')
    parser.add_argument('--assay-csv', type=Path, required=True, help='Evaluator-only frozen assay table')
    parser.add_argument('--dca-features', type=Path, required=True)
    features = parser.add_mutually_exclusive_group(required=True)
    features.add_argument('--tokens', type=Path)
    features.add_argument('--task-features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = ROOT / 'manifests' / (args.cohort + '.json')
    matches = [e for e in json.loads(manifest.read_text())['entries']
               if (e['benchmark'], e['group'], e['method']) == (args.benchmark, args.group, args.method)]
    if len(matches) != 1:
        parser.error('Require exactly one matching frozen run')
    entry = matches[0]
    identity = json.loads((ROOT / 'configs/DATA_IDENTITY.json').read_text())[args.benchmark]
    expected_sha = identity['source_artifacts']['fitness_csv']['sha256']
    if sha256(args.assay_csv) != expected_sha:
        raise ValueError('Frozen assay identity mismatch')
    methods = {'BATA': 'bata', 'ALDE': 'alde', 'RF': 'rf_onehot',
               'EVOLVEpro-650M': 'evolvepro650', 'REAP100-650M': 'reap100',
               'Global-MSE': 'global_mse', 'Equal-rank': 'equal_rank',
               'Evolution-only': 'evolution_only', 'Matched-fine': 'matched_fine',
               'Fine-only-M20': 'fine_m20', 'G20': 'g20', 'G30': 'g30', 'G40': 'g40'}
    method = 'bata' if args.cohort == 'matched_objective24' else methods[args.method]
    objective = args.method if args.cohort == 'matched_objective24' else 'T-DCG-96'
    args.output.mkdir(parents=True, exist_ok=False)
    task_flag = '--tokens' if args.tokens is not None else '--task-features'
    task_path = args.tokens if args.tokens is not None else args.task_features

    def select(round_number, observed):
        feedback = args.output / f'observed_{round_number}.csv'
        with feedback.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['index', 'fitness'])
            writer.writerows(observed.items())
        output = args.output / f'batch_{round_number}.json'
        command = [sys.executable, '-m', 'bata', '--benchmark', args.benchmark,
                   '--method', method, '--objective', objective,
                   '--algorithm-seed', str(entry['algorithm_seed']), '--round', str(round_number),
                   '--observed', str(feedback.resolve()), '--output', str(output.resolve()),
                   '--dca-features', str(args.dca_features.resolve()), task_flag, str(task_path.resolve())]
        subprocess.run(command, cwd=ROOT, check=True)
        return json.loads(output.read_text())['selected_indices']

    def reveal(batch):
        if sha256(args.assay_csv) != expected_sha:
            raise ValueError('Assay changed during replay')
        wanted, values = set(batch), {}
        field = 'fitness' if args.benchmark in ('GB1', 'TrpB') else 'DMS_score'
        with args.assay_csv.open(newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if i in wanted:
                    values[i] = float(row[field])
        return values

    metrics = closed_loop(entry['initial_indices'], identity['candidate_count'], select, reveal)
    metrics.update(cohort=args.cohort, benchmark=args.benchmark, group=args.group,
                   method=args.method, algorithm_seed=entry['algorithm_seed'],
                   manifest_sha256=sha256(manifest), assay_sha256=expected_sha,
                   dca_sha256=sha256(args.dca_features), task_sha256=sha256(task_path))
    with (args.output / 'metrics.json').open('x') as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
