from pathlib import Path
import sys,copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jarvis_hpsafe import bootstrap,safety
from jarvis_hpsafe.common import fold
from jarvis_hpsafe.model import CrossExpertProposalModel
from jarvis_hpsafe.train import model_config,safety_config
from jarvis_hpsafe.inputs import Inputs
from jarvis_hpsafe.metrics import score_predictions

def batch():
    torch.manual_seed(3);tasks=('a','b','c','d','e')
    return tasks,dict(task_index=torch.zeros(7,dtype=torch.long),anchor_standard=torch.randn(7),hidden={t:torch.randn(7,8) for t in tasks},prediction_standard={t:torch.randn(7) for t in tasks},uncertainty_standard={t:torch.zeros(7) for t in tasks},available=torch.ones(7,5,dtype=torch.bool))

def test_no_cross_isolates_reliability_and_all_predictions():
    tasks,b=batch();m=CrossExpertProposalModel(model_config(tasks,{t:8 for t in tasks},True)).eval();b['available'][:,1:]=False
    with torch.inference_mode():
        before=m(**b);changed=copy.deepcopy(b)
        for t in tasks[1:]:
            changed['hidden'][t]*=100;changed['prediction_standard'][t]+=1000;changed['uncertainty_standard'][t]+=500
        after=m(**changed)
    for key in ['raw_standard','data_standard','physics_standard','route_probabilities','data_attention','physics_attention','reliability']:
        assert torch.equal(getattr(before,key),getattr(after,key)),key
    assert before.reliability[:,2].eq(1/5).all()
    assert before.data_attention[:,1:].eq(0).all()

def test_full_uses_available_sources():
    tasks,b=batch();m=CrossExpertProposalModel(model_config(tasks,{t:8 for t in tasks})).eval()
    with torch.inference_mode():
        a=m(**b);b['hidden']['b']+=10;b['prediction_standard']['b']+=10;c=m(**b)
    assert not torch.equal(a.reliability,c.reliability)
    assert not torch.equal(a.data_attention,c.data_attention)
    assert a.data_attention[:,0].eq(0).all()

def test_role_specific_inputs_and_anchor_normalization():
    x=Inputs.__new__(Inputs);x.tasks=('a','b');x.records=[{'metric':'MAE'},{'metric':'ACC'}];ids=['JVASP-1','JVASP-2'];x.lookup={s:i for i,s in enumerate(ids)};x.assignment=np.array([fold(s) for s in ids]);x.dims={'a':2,'b':2};x.stats={'a':(10,2),'b':(0,1)}
    x.prediction={'a':{f:np.array([12+f,14+f],dtype=float) for f in [-1,0,1,2,3,4]},'b':{f:np.array([.2,.8]) for f in [-1,0,1,2,3,4]}}
    x.hidden={t:{f:np.full((2,2),f,dtype=np.float32) for f in [-1,0,1,2,3,4]} for t in x.tasks};x.token_stats={'a':{'a':{'mean':1.,'scale':2.},'b':{'mean':0.,'scale':3.}}}
    b=x.batch('a',ids,'train','cpu',no_cross=True)
    expected=np.array([(x.prediction['a'][fold(s)][i]-10)/2 for i,s in enumerate(ids)],dtype=np.float32)
    assert np.array_equal(b['anchor_standard'].numpy(),expected)
    assert np.array_equal(b['prediction_standard']['a'].numpy(),(expected-1)/2)
    assert b['available'].tolist()==[[True,False],[True,False]]
    full=x.batch('a',ids,'test','cpu');assert np.array_equal(full['anchor_standard'].numpy(),(x.prediction['a'][-1]-10)/2)
    assert x.source_available('a','b',ids,np.array([-1,-1])).all()
    assert np.array_equal(x.anchor('a',ids,'test'),x.prediction['a'][-1])

def test_bootstrap_retains_class_proportions():
    y=np.array([0]*90+[1]*10);anchor=np.array([.1]*80+[.6]*10+[.7]*10);model=np.array([.1]*90+[.7]*10);cfg=safety_config();cfg.update(bootstrap_resamples=40,bootstrap_max_rows=50)
    seed=123;o=bootstrap.class_proportional_bootstrap(task_type='classification',y=y,anchor=anchor,model=model,config=cfg,seed=seed)
    rng=np.random.default_rng(seed);g=[]
    for _ in range(40):
        idx=np.r_[rng.choice(np.arange(90),45,replace=True),rng.choice(np.arange(90,100),5,replace=True)]
        g.append(safety.normalized_gain('classification',score_predictions('classification',y[idx],anchor[idx]),score_predictions('classification',y[idx],model[idx])))
    assert o['lower_gain_percent']==np.quantile(g,.05)
    assert o['upper_gain_percent']==np.quantile(g,.95)
    assert safety._bootstrap_gain is bootstrap.class_proportional_bootstrap

def test_classification_threshold_and_exact_fallback():
    assert score_predictions('classification',[0,1],[.5,.50000001])==1
    anchor=np.array([.125,1.75,-2.0]);raw=np.array([1.3,-.2,4.])
    pred,accepted=safety._apply_parameters(anchor=anchor,raw=raw,probability=np.zeros(3),alpha=.75,threshold=.5)
    assert not accepted.any() and np.array_equal(pred,anchor)
