#!/usr/bin/env python3
"""Recompute No-cross-expert comparisons from the released paired fold scores.

Rebuild No-cross-expert comparison fields while retaining Full-only statistics
and source-use diagnostics.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['expt_is_metal','glass','mp_is_metal','dielectric','expt_gap','jdft2d',
         'log_gvrh','log_kvrh','mp_e_form','mp_gap','perovskites','phonons','steels']
SEED = 20260930
N_BOOTSTRAP = 200_000


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def bootstrap(matrix, statistic='mean', rng=None):
    rng = np.random.default_rng(SEED) if rng is None else rng
    tasks, folds = matrix.shape
    values = np.empty(N_BOOTSTRAP)
    for start in range(0, N_BOOTSTRAP, 5000):
        size = min(5000, N_BOOTSTRAP-start)
        ti = rng.integers(0, tasks, size=(size, tasks))
        fi = rng.integers(0, folds, size=(size, tasks, folds))
        means = np.take_along_axis(matrix[ti], fi, axis=2).mean(axis=2)
        values[start:start+size] = means.mean(axis=1) if statistic == 'mean' else np.median(means, axis=1)
    return values


def compare(actual, expected, location='value'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'{location}: keys differ')
        for key in expected:
            compare(actual[key], expected[key], f'{location}/{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{location}: lengths differ')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{location}/{i}')
    else:
        try:
            a, b = float(actual), float(expected)
        except (ValueError, TypeError):
            if actual != expected:
                raise AssertionError(f'{location}: {actual!r} != {expected!r}')
        else:
            if not np.isclose(a, b, atol=1e-9, rtol=1e-10, equal_nan=True):
                raise AssertionError(f'{location}: {a} != {b}')


def emit(relative, rows, write):
    path = ROOT/relative
    if write:
        if isinstance(rows, list):
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
        else:
            path.write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    else:
        compare(read_csv(path) if isinstance(rows, list) else json.loads(path.read_text()), rows, relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    fold_rows = read_csv(ROOT/'analysis/routing_and_benefit/tables/outer_fold_scores.csv')
    def matrix(variant, field):
        return np.array([[float(next(r[field] for r in fold_rows if r['task']==t
                                    and int(r['outer_fold'])==k and r['variant']==variant))
                          for k in range(5)] for t in TASKS])
    anchor = matrix('Stage1','score'); full = matrix('HP-SafeMoE','score')
    control = matrix('No cross-expert','score')
    F = matrix('HP-SafeMoE','normalized_gain_percent')
    C = matrix('No cross-expert','normalized_gain_percent'); D = F-C
    reg = np.array([next(r['metric'] for r in fold_rows if r['task']==t).startswith('MAE') for t in TASKS])
    calculated = np.where(reg[:,None],100*(anchor-control)/anchor,100*(control-anchor)/(1-anchor))
    np.testing.assert_allclose(C,calculated,atol=1e-9,rtol=1e-10)
    full_native = np.where(reg[:,None],anchor-full,full-anchor)
    delta_native = np.where(reg[:,None],control-full,full-control)
    # Task tables and the mixed source-use summary preserve Full-only fields.
    variants = read_csv(ROOT/'results/matbench/task_variant_mean_std.csv')
    for row in variants:
        if row['variant']=='No cross-expert':
            i=TASKS.index(row['task'])
            row.update(score_mean=control[i].mean(),score_std=control[i].std(ddof=1),
                       gain_mean_percent=C[i].mean(),gain_std_percent=C[i].std(ddof=1))
    emit('results/matbench/task_variant_mean_std.csv',variants,args.write)
    matched = read_csv(ROOT/'results/matbench/task_matched_comparisons.csv')
    for row in matched:
        i=TASKS.index(row['task'])
        row.update(gain_no_cross_expert=C[i].mean(),cross_expert_increment_percent=D[i].mean())
    emit('results/matbench/task_matched_comparisons.csv',matched,args.write)
    sharing=read_csv(ROOT/'analysis/routing_and_benefit/tables/task_correction_and_source_summary.csv')
    for row in sharing:
        row['full_minus_no_cross_expert_percentage_points']=D[TASKS.index(row['task'])].mean()
    emit('analysis/routing_and_benefit/tables/task_correction_and_source_summary.csv',sharing,args.write)
    boot=bootstrap(D);low,high=np.quantile(boot,[.025,.975]);means=D.mean(axis=1)
    signs=np.array(list(itertools.product([-1.,1.],repeat=len(TASKS))))
    sign_p=((signs@means/len(TASKS))>=means.mean()-1e-12).mean()
    aggregate=read_csv(ROOT/'source_data/matbench_aggregate_uncertainty.csv')
    row=next(r for r in aggregate if r['comparison']=='Full HP-SafeMoE vs No-cross-expert')
    row.update(macro_effect_percentage_points=D.mean(),bootstrap_ci95_low=low,bootstrap_ci95_high=high,
               bootstrap_probability_effect_le_zero=(boot<=0).mean(),exact_one_sided_task_signflip_p=sign_p)
    emit('source_data/matbench_aggregate_uncertainty.csv',aggregate,args.write)
    altF=F.copy();altD=D.copy()
    altF[~reg]=100*(full[~reg]-anchor[~reg])/anchor[~reg]
    altD[~reg]=100*(full[~reg]-control[~reg])/anchor[~reg]
    analyses=[(F,D,'mean'),(F[reg],D[reg],'mean'),(F[~reg],D[~reg],'mean'),
              (F,D,'median'),(altF,altD,'mean'),(100*full_native[~reg],100*delta_native[~reg],'mean')]
    sensitivity=read_csv(ROOT/'source_data/gain_normalization_sensitivity.csv')
    rng=np.random.default_rng(SEED)
    for row,(fm,dm,statistic) in zip(sensitivity,analyses):
        bootstrap(fm,statistic,rng)  # Keep the declared paired-comparison RNG schedule.
        b=bootstrap(dm,statistic,rng);lo,hi=np.quantile(b,[.025,.975])
        row.update(full_vs_no_cross=dm.mean() if statistic=='mean' else np.median(dm.mean(axis=1)),
                   full_vs_no_cross_ci95_low=lo,full_vs_no_cross_ci95_high=hi)
    sensitivity[0].update(full_vs_no_cross_ci95_low=low,full_vs_no_cross_ci95_high=high)
    emit('source_data/gain_normalization_sensitivity.csv',sensitivity,args.write)
    native=read_csv(ROOT/'source_data/native_metric_changes.csv')
    for row in native:
        row['full_vs_no_cross_native_improvement']=delta_native[TASKS.index(row['task'])].mean()
    emit('source_data/native_metric_changes.csv',native,args.write)
    mechanism=read_csv(ROOT/'source_data/mechanism_uncertainty.csv')
    for i,row in enumerate(mechanism):
        values=D[TASKS.index(row['target_task'])]
        rng=np.random.default_rng(SEED+503+i*1009)
        lo,hi=np.quantile(values[rng.integers(0,5,size=(100_000,5))].mean(axis=1),[.025,.975])
        row.update(full_minus_no_cross_increment_pp=values.mean(),full_minus_no_cross_ci95_low_pp=lo,
                   full_minus_no_cross_ci95_high_pp=hi,fold_full_minus_no_cross_increments_pp=';'.join(f'{v:.9g}' for v in values),
                   folds_with_positive_full_minus_no_cross_increment=int((values>0).sum()),
                   folds_with_estimable_weight_benefit_rho=int(np.isfinite([
                       float(v) for v in row['fold_weight_benefit_rhos'].split(';')
                   ]).sum()))
    emit('source_data/mechanism_uncertainty.csv',mechanism,args.write)
    native_means=delta_native.mean(axis=1)
    print(json.dumps({'status':'written' if args.write else 'verified','tasks':13,'folds':5,
                      'full_vs_no_cross':{'wins':int((native_means>1e-12).sum()),'ties':int((abs(native_means)<=1e-12).sum()),
                                          'losses':int((native_means< -1e-12).sum())},
                      'increment_pp':float(D.mean()),'ci95':[float(low),float(high)]},indent=2))


if __name__=='__main__':
    main()
