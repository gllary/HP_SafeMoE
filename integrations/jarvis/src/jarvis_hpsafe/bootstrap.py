import numpy as np
from . import safety
from .metrics import score_predictions
REGRESSION_BOOTSTRAP=safety._bootstrap_gain

def class_proportional_bootstrap(*,task_type,y,anchor,model,config,seed):
    if task_type!='classification':
        return REGRESSION_BOOTSTRAP(task_type=task_type,y=y,anchor=anchor,model=model,config=config,seed=seed)
    y,anchor,model=map(np.asarray,(y,anchor,model));counts=np.array([(y==c).sum() for c in (0,1)])
    total=min(len(y),int(config['bootstrap_max_rows']))
    n0=int(round(total*counts[0]/len(y)));sizes=(n0,total-n0)
    assert all(n==0 or s>0 for n,s in zip(counts,sizes))
    def gain(idx):
        return safety.normalized_gain(task_type,score_predictions(task_type,y[idx],anchor[idx]),score_predictions(task_type,y[idx],model[idx]))
    rng=np.random.default_rng(seed);g=[]
    for _ in range(config['bootstrap_resamples']):
        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),size=sizes[c],replace=True) for c in (0,1)])
        g.append(gain(idx))
    confidence=config['bootstrap_confidence']
    return dict(point_gain_percent=gain(np.arange(len(y))),lower_gain_percent=float(np.quantile(g,1-confidence)),
                upper_gain_percent=float(np.quantile(g,confidence)))

safety._bootstrap_gain=class_proportional_bootstrap
