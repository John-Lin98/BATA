"""Aggregate complete replay metrics without mixing cohorts or duplicate runs."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def aggregate(paths):
    rows, identities = [], set()
    for path in paths:
        row = json.loads(Path(path).read_text())
        key = tuple(row[k] for k in ('cohort', 'benchmark', 'method', 'group'))
        if key in identities:
            raise ValueError('Duplicate run identity')
        identities.add(key)
        h = row['history']
        if (row['queries'] != 480 or row['unique_queries'] != 480 or len(h) != 5
                or not all(math.isfinite(v) for v in h) or h != sorted(h)
                or row['Final'] != h[-1]
                or row['Query_AUC'] != (h[0] + 2 * sum(h[1:4]) + h[4]) / 8):
            raise ValueError('Invalid complete-run metrics')
        rows.append({k: row[k] for k in ('cohort', 'benchmark', 'method', 'group', 'algorithm_seed', 'Final', 'Query_AUC')})
    if not rows or len({r['cohort'] for r in rows}) != 1:
        raise ValueError('Exactly one nonempty cohort required')
    rows.sort(key=lambda r: (r['benchmark'], r['method'], r['group']))
    summary = []
    for benchmark, method in sorted({(r['benchmark'], r['method']) for r in rows}):
        group = [r for r in rows if (r['benchmark'], r['method']) == (benchmark, method)]
        cell = dict(cohort=group[0]['cohort'], benchmark=benchmark, method=method, n=len(group))
        for metric in ['Final', 'Query_AUC']:
            values = [r[metric] for r in group]
            cell[metric + '_mean'] = mean(values)
            cell[metric + '_sd'] = stdev(values) if len(values) > 1 else ''
        summary.append(cell)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, summary = aggregate(args.metrics)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, data in [('per_run.csv', rows), ('summary.csv', summary)]:
        with (args.output / filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print('Aggregated supplied runs only; check cohort coverage before paper use')


if __name__ == '__main__':
    main()
