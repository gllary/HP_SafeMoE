#!/usr/bin/env python3
"""Verify all 195 published predictions against separate evaluation targets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {'Stage1':'stage1_specialist_only','No cross-expert':'ablation_no_cross_expert__oof_locked',
            'HP-SafeMoE':'final__oof_locked'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    result = root/'results/matbench'
    with (root/'analysis/routing_and_benefit/tables/outer_fold_scores.csv').open() as handle:
        scores=list(csv.DictReader(handle))
    run=json.loads((result/'experiment/run_record.json').read_text())
    summary=json.loads((result/'experiment/result_summary.json').read_text())
    if summary['source_summary_sha256'] != run['source_artifacts']['result_summary']['sha256']:
        raise AssertionError('Result-summary source hashes differ')
    compatibility=json.loads((result/'experiment/implementation_verification.json').read_text())
    if compatibility['public_model_sha256'] != sha256(root/'src/hpsafe_sota/stage2/model.py'):
        raise AssertionError('Public model differs from the compatibility-verified source')
    commit=json.loads((result/'experiment/prediction_commit.json').read_text())
    if not (commit['committed_unix'] < commit['scored_unix']):
        raise AssertionError('Prediction commitment must precede scoring')
    if len(commit['predictions'])!=65 or run['model_count']!=50:
        raise AssertionError('Incomplete experiment inventory')
    checkpoints=json.loads((result/'experiment/checkpoint_hashes.json').read_text())
    if len(checkpoints)!=50 or len({x['path'] for x in checkpoints})!=50:
        raise AssertionError('Incomplete checkpoint hash inventory')
    for item in commit['predictions']:
        if sha256(root/item['path'])!=item['sha256']:
            raise AssertionError(f"Prediction hash mismatch: {item['path']}")
    target_records=json.loads((result/'experiment/evaluation_targets.json').read_text())
    if len(target_records)!=13:
        raise AssertionError('Expected thirteen target datasets')
    order=['steels','expt_gap','glass','expt_is_metal','jdft2d','dielectric','log_kvrh',
           'log_gvrh','perovskites','phonons','mp_gap','mp_is_metal','mp_e_form']
    checked=0
    means={}
    for item in target_records:
        task=item['task']; target_path=root/item['evaluation_targets']
        if sha256(target_path)!=item['sha256']:
            raise AssertionError(f'Evaluation target hash mismatch: {task}')
        with np.load(target_path,allow_pickle=False) as archive:
            sample_ids=archive['sample_ids'].astype(str);targets=archive['targets'];outer=archive['outer_fold']
        task_scores={name:[] for name in VARIANTS}
        for fold in range(5):
            mask=outer==fold;expected_ids=sample_ids[mask];y=targets[mask]
            for name,variant in VARIANTS.items():
                row=next(x for x in scores if x['task']==task and int(x['outer_fold'])==fold and x['variant']==name)
                with np.load(result/f'per_sample_predictions/outer_{fold}/{variant}/{task}.npz',allow_pickle=False) as a:
                    if 'y_true' in a.files or 'targets' in a.files:
                        raise AssertionError('Prediction files must remain target-free')
                    np.testing.assert_array_equal(a['sample_ids'].astype(str),expected_ids)
                    if len(y)!=int(row['n_samples']):
                        raise AssertionError('Evaluation count mismatch')
                    prediction=a['prediction'];anchor=a['anchor_prediction']
                    np.testing.assert_array_equal(prediction[~a['accepted']],anchor[~a['accepted']])
                    metric=mean_absolute_error if row['metric'].startswith('MAE') else roc_auc_score
                    value=metric(y,prediction)
                    np.testing.assert_allclose(value,float(row['score']),atol=1e-12,rtol=0)
                    if name=='No cross-expert':
                        accepted=summary['folds'][fold]['results'][variant][task]
                        np.testing.assert_allclose(value,accepted['model_score'],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(float(row['normalized_gain_percent']),
                                                   accepted['normalized_gain_percent'],atol=1e-12,rtol=0)
                        index=order.index(task)
                        for field in ['data_attention','learned_relation_attention']:
                            attention=a[field]
                            if np.any(np.delete(attention,index,axis=1)!=0):
                                raise AssertionError(f'Non-target attention in No-cross-expert control: {task}/{fold}')
                    task_scores[name].append(float(value));checked+=1
        means[task]={k:float(np.mean(v)) for k,v in task_scores.items()}
        np.testing.assert_allclose(means[task]['No cross-expert'],
                                   summary['aggregate']['tasks'][task]['model_score'],atol=1e-12,rtol=0)
    wins=ties=losses=0
    for task,v in means.items():
        sign=1 if task in ['glass','expt_is_metal','mp_is_metal'] else -1
        d=sign*(v['HP-SafeMoE']-v['No cross-expert'])
        wins+=int(d>1e-12);losses+=int(d< -1e-12);ties+=int(abs(d)<=1e-12)
    if (wins,ties,losses)!=(9,2,2):
        raise AssertionError(f'Full versus No-cross-expert count mismatch: {(wins,ties,losses)}')
    return {'status':'verified','task_fold_variant_checks':checked,'no_cross_prediction_hashes':65,
            'full_vs_no_cross':{'wins':wins,'ties':ties,'losses':losses}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify the committed release without changing it.')
    parser.parse_args()
    print(json.dumps(verify(),indent=2))


if __name__=='__main__':
    main()
