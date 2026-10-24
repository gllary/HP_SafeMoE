#!/usr/bin/env python3
"""Independently verify released JARVIS scores and exact anchor fallback."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def verify():
    source=json.loads((ROOT/'experiment_results.json').read_text());rows=source['tasks']
    proof=json.loads((ROOT/'prediction_provenance.json').read_text());assert len(rows)==34
    summary=pd.read_csv(ROOT/'task_metrics.csv',float_precision='round_trip').set_index('task')
    total=0;comparisons=0;directions={k:[] for k in ['stage1_full_test','no_cross_expert_stage2_test','public_best']}
    for r in rows:
        t=r['task'];p=ROOT/'per_sample_predictions'/f'{t}.csv'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['tasks'][t]['csv_sha256'],t
        x=pd.read_csv(p,float_precision='round_trip');assert len(x)==r['n_test']==proof['tasks'][t]['rows'] and x.sample_id.is_unique
        total+=len(x);y=x.y_true.to_numpy();anchor=x.stage1_anchor_prediction.to_numpy()
        def score(p):return float(np.mean((p>.5)==y)) if r['metric']=='ACC' else float(np.mean(np.abs(p-y)))
        assert abs(score(anchor)-r['stage1_full_test'])<1e-12
        for variant,col in [('full','full'),('no_cross_expert','no_cross')]:
            pred=x[col+'_prediction'].to_numpy();raw=x[col+'_raw_prediction'].to_numpy();accepted=x[col+'_accepted'].to_numpy(dtype=bool);alpha=x[col+'_residual_scale'].to_numpy()
            assert np.array_equal(pred[~accepted],anchor[~accepted]),(t,col,'fallback')
            reconstructed=anchor+accepted*alpha*(raw-anchor)
            assert np.allclose(pred,reconstructed,atol=1e-12,rtol=0),(t,col,'residual equation')
            assert not np.any(accepted & ~x[col+'_task_certificate_accepted'].to_numpy(dtype=bool))
            value=score(pred);key=variant+'_stage2_test';assert abs(value-r[key])<1e-12,(t,col,value,r[key])
            assert abs(value-summary.loc[t,key])<1e-12
            assert abs(float(accepted.mean())-r[variant+'_coverage'])<1e-12
            comparisons+=1
        for k in directions:
            delta=(r[k]-r['full_stage2_test'])*(1 if r['metric']=='MAE' else -1)
            directions[k].append(0 if abs(delta)<=1e-12 else 1 if delta>0 else -1)
    assert total==95477==proof['test_rows']
    counts={k:[v.count(1),v.count(0),v.count(-1)] for k,v in directions.items()}
    assert list(counts.values())==[[25,8,1],[24,5,5],[23,1,10]]
    cert=json.loads((ROOT/'protocol_certificate.json').read_text());assert cert['task_count']==34 and cert['prediction_barrier_before_test_scoring']
    return {'tasks':34,'test_rows':total,'final_scores_checked':comparisons,'exact_anchor_fallback':True,'comparisons':counts}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.parse_args()
    print(json.dumps(verify(),indent=2))
