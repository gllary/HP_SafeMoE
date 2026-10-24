"""Verify frozen external correction decisions and recompute source shares."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify(write=False):
    tasks = yaml.safe_load((ROOT/'configs/experiments/stage2.yaml').read_text())['task_order']
    acceptance = pd.read_csv(ROOT/'source_data/external_correction_acceptance.csv').set_index(['dataset','outer_fold'])
    shares = []
    for dataset, target in [('DS2-248','expt_gap'),('Core-23','dielectric')]:
        frame = pd.read_csv(ROOT/f'results/{dataset}/outputs/ensemble_predictions.csv')
        provenance = json.loads((ROOT/f'results/{dataset}/experiment/prediction_provenance.json').read_text())
        total_weights = np.zeros(len(tasks))
        for record in provenance['records']:
            fold = record['outer_fold']
            path = ROOT/f'results/{dataset}/experiment'/record['export_path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['export_sha256']
            with np.load(path,allow_pickle=False) as a:
                assert np.array_equal(a['sample_ids'].astype(str),frame.sample_id.astype(str))
                for key,column in [('anchor_prediction',f'stage1_outer_{fold}'),('prediction',f'stage2_outer_{fold}')]:
                    np.testing.assert_allclose(a[key],frame[column],atol=1e-12,rtol=0)
                accepted = a['accepted'].astype(bool)
                row = acceptance.loc[(dataset,fold)]
                assert row.n_samples == len(accepted) and row.n_accepted == accepted.sum()
                assert abs(row.sample_acceptance_fraction - accepted.mean()) < 1e-12
                weights = (a['route_probabilities'][:,1,None]*a['data_attention']
                           + a['route_probabilities'][:,2,None]*a['learned_relation_attention'])
                weights[:,tasks.index(target)] = 0
                total_weights += weights[accepted].sum(axis=0)
        for source, value in zip(tasks,total_weights / total_weights.sum() * 100):
            if source != target:
                shares.append({'dataset':dataset,'target_task':target,'source_expert':source,
                               'conditional_source_weight_percent':value})
    actual = pd.DataFrame(shares)
    path = ROOT/'source_data/external_source_weights.csv'
    if write:
        actual.to_csv(path,index=False,lineterminator='\n')
    else:
        expected = pd.read_csv(path)
        assert actual.iloc[:,:3].equals(expected.iloc[:,:3])
        np.testing.assert_allclose(actual.iloc[:,3],expected.iloc[:,3],atol=1e-10,rtol=0)
    return {'diagnostic_archives':10, 'source_share_rows':len(actual),
            'mean_acceptance_percent':(acceptance.groupby('dataset').sample_acceptance_fraction.mean()*100).to_dict()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(write=args.write),indent=2))
