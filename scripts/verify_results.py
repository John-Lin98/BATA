"""Verify published file digests and reproduce main/sensitivity summary cells."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]/'results'
def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def verify_calibration():
    """Frozen 24-pair percentile bootstrap; descriptive, not multiplicity-adjusted."""
    folder = ROOT/'calibration'
    objectives = ('G-Rank', 'T-Uniform-96', 'T-DCG-96', 'T-DCG-192')
    benchmarks = ('GB1', 'PABP', 'TrpB')
    data = rows(folder/'matched_objective_per_run.csv')
    lookup = {(r['benchmark'], r['objective'], int(r['group'])): r for r in data}
    expected_keys = {(b, o, g) for b in benchmarks for o in objectives for g in range(24)}
    assert len(data) == len(lookup) == 288 and set(lookup) == expected_keys
    for b in benchmarks:
        for g in range(24):
            pair = [lookup[b, o, g] for o in objectives]
            for field in ('algorithm_seed', 'manifest_sha256', 'initial96_sha256'):
                assert len({r[field] for r in pair}) == 1, (b, g, field)
            assert all(int(r['unique_queries']) == 480 for r in pair)
    summaries = rows(folder/'matched_objective_summary.csv')
    effects = rows(folder/'B96_paired_effects.csv')
    ranks = rows(folder/'mean_rank.csv')
    assert len(summaries) == 12 and len(effects) == 18 and len(ranks) == 4
    def close(actual, expected):
        assert np.isclose(actual, float(expected), rtol=0, atol=1e-14), (actual, expected)
    computed_ranks = {}
    for bi, b in enumerate(benchmarks):
        for mi, metric in enumerate(('Final', 'Query_AUC')):
            arrays = {o: np.array([float(lookup[b, o, g][metric]) for g in range(24)])
                      for o in objectives}
            rank = rankdata(-np.array([arrays[o].mean() for o in objectives]), method='average')
            draw = np.random.default_rng(np.random.SeedSequence([20260916, bi, mi])).integers(
                0, 24, size=(20000, 24))
            for o, place in zip(objectives, rank):
                matches = [r for r in summaries if r['benchmark'] == b and r['objective'] == o]
                assert len(matches) == 1 and int(matches[0]['n']) == 24
                cell = matches[0]; values = arrays[o]
                q1, median, q3 = np.quantile(values, [.25, .5, .75])
                for field, value in dict(mean=values.mean(), sd=values.std(ddof=1),
                                         q1=q1, median=median, q3=q3, iqr=q3-q1, rank=place).items():
                    close(value, cell[metric+'_'+field])
                computed_ranks.setdefault((o, metric), []).append(place)
            for other in ('G-Rank', 'T-Uniform-96', 'T-DCG-192'):
                matches = [r for r in effects if r['benchmark'] == b and r['metric'] == metric
                           and r['lhs'] == 'T-DCG-96' and r['rhs'] == other]
                assert len(matches) == 1
                cell = matches[0]
                delta = arrays['T-DCG-96'] - arrays[other]
                low, high = np.quantile(delta[draw].mean(axis=1), [.025, .975])
                for field, value in dict(paired_mean_difference=delta.mean(), ci95_low=low,
                                         ci95_high=high, n_pairs=24, positive_pairs=(delta>0).sum(),
                                         equal_pairs=(delta==0).sum(), negative_pairs=(delta<0).sum()).items():
                    close(value, cell[field])
    assert {r['objective'] for r in ranks} == set(objectives)
    for cell in ranks:
        assert int(cell['tasks']) == 3
        for metric in ('Final', 'Query_AUC'):
            close(np.mean(computed_ranks[cell['objective'], metric]), cell[metric+'_mean_rank'])
    return len(summaries), len(effects)

def verify_variants():
    """Reaggregate independent variant cohorts without pooling development data."""
    method_names = {'bata': 'BATA', 'fine_m20': 'Fine-only-M20',
                    'g20': 'G20', 'g30': 'G30', 'g40': 'G40'}
    def group(data, benchmark, method, n):
        found = [r for r in data if r['benchmark'] == benchmark and r['method'] == method]
        assert len(found) == n and len({r['group'] for r in found}) == n, (benchmark, method)
        return {metric: np.array([float(r[metric]) for r in found])
                for metric in ('Final', 'Query_AUC')}
    def close(actual, expected):
        assert np.isclose(actual, expected, rtol=0, atol=1e-14), (actual, expected)
    cells = 0
    for cohort, summary_name, n, methods in (
        ('gate_development5', 'gate_development.json', 5, tuple(method_names)),
        ('g20_confirmation10', 'g20_confirmation.json', 10, ('bata', 'fine_m20', 'g20')),
    ):
        data = rows(ROOT/'variants'/f'{cohort}_per_run.csv')
        expected = json.loads((ROOT/'variants'/summary_name).read_text())['summary']
        assert len(data) == len(expected)*len(methods)*n
        assert {r['cohort'] for r in data} == {cohort}
        for benchmark, arms in expected.items():
            for method in methods:
                values = group(data, benchmark, method_names[method], n)
                cell = arms[method]
                for metric, field in (('Final', 'Final'), ('Query_AUC', 'AUC')):
                    suffix = '_mean' if cohort == 'g20_confirmation10' else ''
                    close(values[metric].mean(), cell[field+suffix])
                    close(values[metric].std(ddof=1), cell[field+'_sd'])
                cells += 1
    formal = rows(ROOT/'variants/fine_m20_formal118_per_run.csv')
    trpb = rows(ROOT/'variants/fine_m20_trpb70_per_run.csv')
    assert len(formal) == 118 and len(trpb) == 140
    assert {r['cohort'] for r in formal} == {'fine_m20_formal118'}
    assert {r['cohort'] for r in trpb} == {'fine_m20_trpb70'}
    expected = json.loads((ROOT/'variants/fine_only_m20.json').read_text())['summary']
    main = rows(ROOT/'main35_per_run.csv')
    transfer = rows(ROOT/'appendix/transfer_per_run.csv')
    aliases = {'ALDE': 'ALDE', 'EVOLVEpro650': 'EVOLVEpro-650M'}
    for benchmark, cell in expected.items():
        data = trpb if benchmark == 'TrpB' else formal
        values = group(data, benchmark, 'Fine-only-M20', cell['n'])
        for metric, field in (('Final', 'Final'), ('Query_AUC', 'AUC')):
            close(values[metric].mean(), cell['Fine_'+field])
            close(values[metric].std(ddof=1), cell['Fine_'+field+'_sd'])
        if benchmark in ('HIS7', 'GRB2'):
            source = [dict(r, method=r['arm']) for r in transfer]
            bata = group(source, benchmark, 'bata', 24)
            baseline = group(source, benchmark,
                             {'RF-OneHot': 'rf_onehot', 'REAP100': 'reap100'}[cell['SOTA_method']], 24)
        else:
            bata = group(main, benchmark, 'BATA', 35)
            baseline = (group(trpb, benchmark, 'REAP100-650M', 70) if benchmark == 'TrpB'
                        else group(main, benchmark, aliases[cell['SOTA_method']], 35))
        # TrpB deliberately compares Fine70 with the frozen BATA35 reference;
        # validate the published cross-cohort arithmetic, not a paired effect.
        for metric, field in (('Final', 'Final'), ('Query_AUC', 'AUC')):
            close(bata[metric].mean(), cell['BATA_'+field])
            close(baseline[metric].mean(), cell['SOTA_'+field])
        close(values['Final'].mean()-bata['Final'].mean(), cell['Delta_vs_BATA_Final'])
        close(values['Final'].mean()-baseline['Final'].mean(), cell['Delta_vs_SOTA_Final'])
        cells += 1
    return cells

def verify_engineering_speed():
    """Check paper timing arithmetic and keep incomplete M100 attempts non-results."""
    data = rows(ROOT/'appendix/speed_comparison.csv')
    matrix = [r for r in data if r['family'] == 'speed_matrix']
    pilot = [r for r in data if r['family'] == 'm100_pilot']
    variants = ('old_m5', 'fast_m5', 'reap10', 'reap100')
    assert len(matrix) == 12 and len(pilot) == 3
    by_key = {(r['variant'], int(r['group'])): r for r in matrix}
    assert set(by_key) == {(variant, group) for variant in variants for group in range(3)}
    for group in range(3):
        old, fast = by_key['old_m5', group], by_key['fast_m5', group]
        assert old['Final'] == fast['Final'] and old['Query_AUC'] == fast['Query_AUC']
    medians = {variant: float(np.median([
        float(by_key[variant, group]['campaign_wall_s']) for group in range(3)]))
        for variant in variants}
    assert round(medians['old_m5'], 2) == 1092.19
    assert round(medians['fast_m5'], 2) == 171.96
    assert round(medians['fast_m5'] / medians['reap10'], 2) == 1.91
    assert all(r['status'] == 'failed' and not r['Final'] and not r['Query_AUC'] for r in pilot)
    assert all(r['engineering_only'] == 'True' and r['performance_selection_evidence'] == 'False'
               for r in data)
    return len(data)

def verify():
    main=json.loads((ROOT/'MAIN_PROVENANCE.json').read_text())
    calibration=json.loads((ROOT/'calibration/PROVENANCE.json').read_text())
    variants=json.loads((ROOT/'variants/PROVENANCE.json').read_text())
    artifacts=main['outputs']+calibration['artifacts']+variants['artifacts']
    variant_runs=json.loads((ROOT/'variants/PER_RUN_PROVENANCE.json').read_text())
    artifacts += [dict(item, file='variants/'+item['file']) for item in variant_runs['artifacts']]
    appendix=json.loads((ROOT/'appendix/PROVENANCE.json').read_text())
    artifacts += [dict(item, file='appendix/'+item['file']) for item in appendix['outputs']]
    for item in artifacts:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    manifests=ROOT.parent/'manifests'
    identities=json.loads((manifests/'PROVENANCE.json').read_text())
    for item in identities['artifacts']:
        path=manifests/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        entries=json.loads(path.read_text())['entries'];assert len(entries)==item['arm_runs']
        common={}
        for entry in entries:
            ids=entry['initial_indices'];assert len(ids)==len(set(ids))==96
            assert not entry['labels_included']
            key=(entry['benchmark'],entry['group'])
            if key in common:assert common[key]==ids
            common[key]=ids
        assert len(common)==item['distinct_initializations']
    for cohort in ('main35','sensitivity70'):
        data=rows(ROOT/(cohort+'_per_run.csv'))
        for cell in rows(ROOT/(cohort+'_summary.csv')):
            group=[r for r in data if r['benchmark']==cell['benchmark'] and r['method']==cell['method']]
            assert len(group)==int(cell['n']) and len({r['group'] for r in group})==len(group)
            for metric in ('Final','Query_AUC'):
                values=np.array([float(r[metric]) for r in group])
                assert np.isclose(values.mean(),float(cell[metric+'_mean']),rtol=0,atol=1e-14)
                assert np.isclose(values.std(ddof=1),float(cell[metric+'_sd']),rtol=0,atol=1e-14)
    extra_cells=0
    for data_name,summary_name,section in [
        ('ablation_per_run.csv','ablation24.json','summary'),
        ('transfer_per_run.csv','transfer24.json',None),
        ('trpb_capacity_development_per_run.csv','trpb_capacity_development24.json','summary')]:
        data=rows(ROOT/'appendix'/data_name)
        expected=json.loads((ROOT/'appendix'/summary_name).read_text())
        if section:expected=expected[section]
        for benchmark,arms in expected.items():
            for arm,cell in arms.items():
                group=[r for r in data if r['benchmark']==benchmark and r['arm']==arm]
                assert len(group)==cell['n'] and len({r['group'] for r in group})==len(group)
                for metric in ('Final','Query_AUC'):
                    values=np.array([float(r[metric]) for r in group])
                    assert np.isclose(values.mean(),cell[metric]['mean'],rtol=0,atol=1e-14)
                    assert np.isclose(values.std(ddof=1),cell[metric]['sd'],rtol=0,atol=1e-14)
                extra_cells+=1
    curves=rows(ROOT/'appendix/curves_per_run.csv')
    for cell in rows(ROOT/'appendix/curves_summary.csv'):
        group=[r for r in curves if all(r[k]==cell[k] for k in ('benchmark','arm','queries'))]
        assert len(group)==int(cell['n']) and len({r['group'] for r in group})==len(group)
        values=np.array([float(r['best_fitness']) for r in group])
        assert np.isclose(values.mean(),float(cell['mean']),rtol=0,atol=1e-14)
        assert np.isclose(values.std(ddof=1),float(cell['sd']),rtol=0,atol=1e-14)
    names={'BATA':'bata','ALDE':'alde','EVOLVEpro-650M':'evolvepro650',
           'REAP100-650M':'reap100','RF':'rf_onehot'}
    endpoints=rows(ROOT/'appendix/transfer_per_run.csv') + [dict(r,arm=names[r['method']])
        for r in rows(ROOT/'main35_per_run.csv')]
    indexed={}
    for r in curves:
        indexed.setdefault((r['benchmark'],r['arm'],r['group']),[]).append(r)
    assert len(indexed)==len(endpoints)==765
    for r in endpoints:
        curve=sorted(indexed[(r['benchmark'],r['arm'],r['group'])],key=lambda c:int(c['queries']))
        queries=np.array([int(c['queries']) for c in curve])
        best=np.array([float(c['best_fitness']) for c in curve])
        assert np.array_equal(queries,[96,192,288,384,480])
        assert np.isfinite(best).all() and (np.diff(best)>=0).all()
        assert np.isclose(best[-1],float(r['Final']),rtol=0,atol=1e-14)
        assert np.isclose(np.trapz(best,queries)/384,float(r['Query_AUC']),rtol=0,atol=1e-14)
    speed_rows = verify_engineering_speed()
    variant_cells = verify_variants()
    calibration_cells, paired_cells = verify_calibration()
    print(f'PASS: {len(artifacts)} result hashes; 25 main + {extra_cells} appendix + {variant_cells} variant mean/SD cells; {calibration_cells} calibration summaries + {paired_cells} paired bootstrap contrasts; {speed_rows} engineering timing rows; 125 curve cells; 765 endpoint/AUC trajectories; {len(identities["artifacts"])} matched initialization manifest sets')
if __name__=='__main__':verify()
