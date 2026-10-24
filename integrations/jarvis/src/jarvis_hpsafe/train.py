"""Joint training, validation calibration and committed test predictions."""
import argparse
from dataclasses import asdict
import json
import math
import os
import random
import sys
import time
import numpy as np
from .common import protocol_hash, verify_source_lock
from .common import ROOT,atomic_json,registry,sha256,manifest,SOURCE_ROOT
from .common import PROTOCOL
from . import bootstrap  # Install class-proportional validation resampling.

def model_config(tasks,dims,no_cross=False):
    from .model import ProposalConfig
    return ProposalConfig(task_names=tasks,expert_ids=tasks,hidden_dims=dims,anchor_indices=tuple(range(len(tasks))),token_dim=128,task_dim=32,residual_hidden=192,residual_rank=16,dropout=.08,attention_top_k=4,residual_limit=1.25,route_temperature=.75,use_cross_expert_input=not no_cross,use_data_branch=True,use_physics_branch=True,use_learned_task_source_relation=True,use_shared_low_rank_basis=False,use_task_private_correction=True,use_sample_gate=True)

def safety_config():
    return dict(meta_crossfit_partitions=5,benefit_gate_max_rows=12000,benefit_gate_learning_rate=.03,benefit_gate_weight_decay=.001,benefit_gate_steps=160,residual_scale_grid=[0.,.25,.5,.75,1.],benefit_probability_threshold_grid=[.35,.45,.55,.65,.75],minimum_sample_coverage=.05,bootstrap_max_rows=4096,bootstrap_resamples=400,bootstrap_confidence=.95)

def predict(inputs,model,task,role,device,jids=None):
    import torch
    keys=['raw_prediction','data_prediction','physics_prediction','route_probabilities','route_index','data_attention','physics_attention','reliability','anchor_prediction']
    parts={k:[] for k in keys};jids=inputs.roles[task][role] if jids is None else jids;model.eval()
    with torch.inference_mode():
        for start in range(0,len(jids),128):
            subset=jids[start:start+128];batch=inputs.batch(task,subset,role,device,no_cross=not model.config.use_cross_expert_input);out=model(**batch)
            for key,value in [('raw_prediction',out.raw_standard),('data_prediction',out.data_standard),('physics_prediction',out.physics_standard)]:parts[key].append(inputs.restore(task,value.cpu().numpy()))
            parts['anchor_prediction'].append(inputs.anchor(task,subset,role))
            for key,value in [('route_probabilities',out.route_probabilities),('route_index',out.route_one_hot.argmax(1)),('data_attention',out.data_attention),('physics_attention',out.physics_attention),('reliability',out.reliability)]:parts[key].append(value.cpu().numpy())
    result={'sample_ids':np.asarray(jids,dtype=str),**{k:np.concatenate(v) for k,v in parts.items()}}
    assert all(np.isfinite(v).all() for k,v in result.items() if k!='sample_ids')
    return result

def normalized_loss(inputs,t,record,role,key='raw_prediction',selection=True):
    y=np.asarray([inputs.labels[t][role][j] for j in record['sample_ids']]);p=record[key]
    if inputs.records[inputs.tasks.index(t)]['metric']=='ACC':
        if not selection:return float(np.mean((p>.5)!=y))
        p=np.clip(p,1e-7,1-1e-7);return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
    return float(np.mean(np.abs(y-p))/inputs.stats[t][1])

def save_torch(path,state):
    import torch
    temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)

def fit_member(inputs,variant,seed,device,lock):
    import torch
    from .model import CrossExpertProposalModel
    root=ROOT/'stage2'/variant;root.mkdir(exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    model=CrossExpertProposalModel(model_config(inputs.tasks,inputs.dims,variant=='no_cross_expert')).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    resume=root/f'seed{seed}_resume.pt';bestpath=root/f'seed{seed}_best.pt';initial=0;best=float('inf');best_step=0
    steps=lock['stage2_steps'];size=lock['batch_per_task'];interval=lock['val_interval_steps']
    if resume.exists():
        s=torch.load(resume,map_location=device,weights_only=False);assert s['protocol_sha256']==protocol_hash()
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);initial=s['step'];best=s['best'];best_step=s['best_step'];rng.bit_generator.state=s['numpy_rng'];torch.set_rng_state(s['torch_rng'].cpu())
        if s['cuda_rng'] is not None:torch.cuda.set_rng_state(s['cuda_rng'].cpu())
    started=time.time();history=[];historyfile=root/f'seed{seed}_history.json'
    if historyfile.exists():history=json.loads(historyfile.read_text())
    trainids={t:list(inputs.labels[t]['train']) for t in inputs.tasks}
    for step in range(initial,steps):
        model.train();opt.zero_grad(set_to_none=True);losses=[];anchors=[];aux=[];entropy=[]
        for task in inputs.tasks:
            ids=trainids[task];idx=rng.choice(len(ids),size,replace=len(ids)<size);selected=[ids[i] for i in idx]
            batch=inputs.batch(task,selected,'train',device,.05,rng,no_cross=variant=='no_cross_expert');out=model(**batch)
            y=np.asarray([inputs.labels[task]['train'][j] for j in selected],dtype=np.float32);classification=inputs.records[inputs.tasks.index(task)]['metric']=='ACC'
            if not classification:y=(y-inputs.stats[task][0])/inputs.stats[task][1]
            target=torch.as_tensor(y,device=device);criterion=torch.nn.functional.binary_cross_entropy_with_logits if classification else torch.nn.functional.l1_loss
            losses.append(criterion(out.raw_standard,target));anchors.append(criterion(batch['anchor_standard'],target).detach());aux.append((criterion(out.data_standard,target)+criterion(out.physics_standard,target))/2)
            probs=out.route_probabilities[:,1:].clamp_min(1e-8);entropy.append(-(probs*probs.log()).sum(1).mean())
        v=torch.stack(losses);regret=torch.relu(v-torch.stack(anchors));cvar=torch.topk(regret,k=math.ceil(.25*len(inputs.tasks))).values.mean();h=torch.stack(entropy).mean()
        objective=v.mean()+.75*regret.mean()+.25*cvar+.20*torch.stack(aux).mean()+.003*(h-.45).square()
        assert torch.isfinite(objective);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if (step+1)%10==0 or step==0:atomic_json(root/f'seed{seed}_progress.json',dict(variant=variant,seed=seed,step=step+1,steps=steps,objective=float(objective.detach()),regret=float(regret.mean().detach()),seconds=time.time()-started,best_validation=best if math.isfinite(best) else None,best_step=best_step))
        if (step+1)%interval==0 or step+1==steps:
            scores={t:normalized_loss(inputs,t,predict(inputs,model,t,'val',device),'val') for t in inputs.tasks};val=float(np.mean(list(scores.values())))
            if val<best:
                best=val;best_step=step+1;save_torch(bestpath,{'model':model.state_dict(),'step':best_step,'validation':scores,'config':asdict(model.config),'protocol_sha256':protocol_hash()})
            history.append({'step':step+1,'mean_normalized_validation_loss':val,'task_scores':scores});atomic_json(historyfile,history)
            save_torch(resume,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,best=best,best_step=best_step,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,protocol_sha256=protocol_hash()))
            print('STAGE2_VALIDATION',variant,seed,step+1,val,'best',best_step,flush=True)
    state=torch.load(bestpath,map_location=device,weights_only=False);model.load_state_dict(state['model']);records={};gaps={}
    for t in inputs.tasks:
        records[t]=predict(inputs,model,t,'val',device)
        path=root/'validation_records'/f'seed{seed}';path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/(t+'.npz'),**records[t])
        allids=trainids[t];rg=np.random.default_rng(20261014);ids=[allids[i] for i in sorted(rg.choice(len(allids),min(2048,len(allids)),replace=False))]
        tr=predict(inputs,model,t,'train',device,jids=ids)
        gaps[t]={'train_rows':len(ids),'train_proposal_loss':normalized_loss(inputs,t,tr,'train'),'train_anchor_loss':normalized_loss(inputs,t,tr,'train',key='anchor_prediction'),'val_proposal_loss':normalized_loss(inputs,t,records[t],'val'),'val_anchor_loss':normalized_loss(inputs,t,records[t],'val',key='anchor_prediction')}
    atomic_json(root/f'seed{seed}_selected.json',dict(seed=seed,best_step=state['step'],checkpoint_sha256=sha256(bestpath),train_val_gaps=gaps,test_labels_read=False))
    return model,records

def calibrate(inputs,variant,records):
    from .safety import _calibrate_task,_apply_calibration
    root=ROOT/'stage2'/variant;certs={};scores={p:[] for p in ['conservative','balanced','liberal']}
    for n,t in enumerate(inputs.tasks):
        y=np.asarray([inputs.labels[t]['val'][j] for j in inputs.roles[t]['val']]);kind='classification' if inputs.records[n]['metric']=='ACC' else 'regression'
        assert len(y)>=10,'Insufficient official validation labels'
        c=_calibrate_task(records=records[t],y_true=y,task_type=kind,config=safety_config(),seed=20261074+n)
        c.update(status='empirical_validation_meta_crossfit',fit_labels='official_validation_only',proposal_fit_labels='official_train_only')
        certs[t]=c
        for profile in scores:
            applied=_apply_calibration(records[t],calibration=c,profile=profile)
            assert np.array_equal(applied['prediction'][~applied['accepted']],applied['anchor_prediction'][~applied['accepted']])
            scores[profile].append(normalized_loss(inputs,t,applied,'val',key='prediction',selection=False))
    means={k:float(np.mean(v)) for k,v in scores.items()};chosen=min(means,key=means.get)
    payload=dict(primary_profile=chosen,global_validation_scores=means,tasks=certs,test_labels_read=False)
    atomic_json(root/'calibration_locked.json',payload);return payload

def canary():
    import copy
    import torch
    from .model import CrossExpertProposalModel
    from .safety import _calibrate_task,_apply_calibration
    tasks=tuple(r['id'] for r in registry()['tasks']);dims={t:128 for t in tasks};torch.manual_seed(17)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=CrossExpertProposalModel(model_config(tasks,dims)).to(device)
    b=dict(task_index=torch.zeros(8,device=device,dtype=torch.long),anchor_standard=torch.zeros(8,device=device),hidden={t:torch.randn(8,128,device=device) for t in tasks},prediction_standard={t:torch.randn(8,device=device) for t in tasks},uncertainty_standard={t:torch.zeros(8,device=device) for t in tasks},available=torch.ones(8,34,device=device,dtype=torch.bool))
    o=model(**b);target=torch.linspace(-1,1,8,device=device);loss=(o.raw_standard-target).square().mean()+(o.data_standard-target).abs().mean();loss.backward();assert torch.isfinite(loss)
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
    own=CrossExpertProposalModel(model_config(tasks,dims,True)).to(device).eval();b['available'][:]=False;b['available'][:,0]=True
    with torch.no_grad():
        first=own(**b).raw_standard;changed=copy.deepcopy(b)
        for t in tasks[1:]:changed['hidden'][t]=torch.randn_like(changed['hidden'][t])*100;changed['prediction_standard'][t]=torch.randn_like(changed['prediction_standard'][t])*100
        assert torch.equal(first,own(**changed).raw_standard),'No-cross reliability path leaked other experts'
    y=np.asarray([0,1]*20);anchor=np.where(y,.7,.3);rec=dict(sample_ids=np.asarray([str(i) for i in range(40)]),anchor_prediction=anchor,raw_prediction=anchor,data_prediction=anchor,physics_prediction=anchor,route_probabilities=np.tile([0.,.5,.5],(40,1)),route_index=np.ones(40),data_attention=np.full((40,34),1/34),physics_attention=np.full((40,34),1/34),reliability=np.zeros((40,8)))
    cfg=safety_config();cfg.update(bootstrap_resamples=10,benefit_gate_steps=2)
    for kind,truth in [('classification',y),('regression',y.astype(float))]:
        cert=_calibrate_task(records=[rec,rec],y_true=truth,task_type=kind,config=cfg,seed=19)
        for profile in ['conservative','balanced','liberal']:
            out=_apply_calibration([rec,rec],calibration=cert,profile=profile);assert np.array_equal(out['prediction'],anchor)
    from .metrics import score_predictions
    assert score_predictions('classification',[0,1],[.5,.6])==1.
    atomic_json(ROOT/'status/stage2_canary.json',dict(passed=True,synthetic_only=True,tasks=34,no_cross_perturbation_invariance=True,bit_exact_fallback=True,classification_tie_checked=True,device=device,test_labels_read=False))
    print('STAGE2_CANARY_PASSED',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--canary',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.5,0)
    if a.canary:canary();return
    assert torch.cuda.is_available()
    run_experiment('cuda')

def run_experiment(device):
    verify_source_lock()
    ready=json.loads((ROOT/'status/preflight.json').read_text())
    assert ready['passed'] is True and ready['source_stage1_fits']==204 and ready['test_labels_read'] is False
    from .inputs import Inputs
    from .safety import _apply_calibration
    lock=PROTOCOL;inputs=Inputs();root=ROOT/'stage2';allmodels={};calibrations={}
    for variant in ['full','no_cross_expert']:
        models=[];records={t:[] for t in inputs.tasks}
        for seed in lock['stage2_seeds']:
            model,recs=fit_member(inputs,variant,seed,device,lock);models.append(model.cpu())
            for t in inputs.tasks:records[t].append(recs[t])
        calibrations[variant]=calibrate(inputs,variant,records);allmodels[variant]=models
    # Seal all decisions for both independently trained variants before test inference/scoring.
    atomic_json(root/'selection_locked.json',dict(protocol_sha256=protocol_hash(),variants={k:sha256(root/k/'calibration_locked.json') for k in calibrations},input_audit_sha256=sha256(root/'input_audit.json'),normalization_sha256=sha256(root/'normalization.json'),source_lock_sha256=sha256(ROOT/'status/source_lock.json'),test_labels_read=False,time=time.time()))
    committed={}
    for variant,models in allmodels.items():
        for model in models:model.to(device).eval()
        committed[variant]={};cal=calibrations[variant]
        for t in inputs.tasks:
            records=[predict(inputs,m,t,'test',device) for m in models];arrays=_apply_calibration(records,calibration=cal['tasks'][t],profile=cal['primary_profile'])
            assert np.array_equal(arrays['prediction'][~arrays['accepted']],arrays['anchor_prediction'][~arrays['accepted']])
            path=root/variant/'test_predictions'/(t+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)
            committed[variant][t]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}
        for model in models:model.cpu()
    atomic_json(root/'predictions_committed.json',dict(variants=committed,selection_sha256=sha256(root/'selection_locked.json'),test_labels_read=False,time=time.time()))
    from .metrics import score_predictions
    results=[]
    for n,t in enumerate(inputs.tasks):
        truth=json.loads((SOURCE_ROOT/'data/tasks'/t/'test.json').read_text());kind='classification' if inputs.records[n]['metric']=='ACC' else 'regression';record=inputs.records[n]
        row=dict(task=t,name=record['name'],metric=record['metric'],unit=record.get('unit'),stage1_model=('Official public ALIGNN checkpoint' if t in inputs.public_overrides else record['model']),public_full_override=t in inputs.public_overrides,public_best=record.get('recomputed_score'),public_model=record.get('public_model',record.get('model')),public_source=record['prediction_url'],n_test=len(truth),split_counts=manifest(t)['official_counts'])
        for variant in committed:
            item=committed[variant][t];assert sha256(ROOT/item['path'])==item['sha256'];arr=np.load(ROOT/item['path']);ids=arr['sample_ids'].astype(str);assert len(ids)==len(truth) and set(ids)==set(truth)
            y=np.asarray([truth[j] for j in ids]);anchor=score_predictions(kind,y,arr['anchor_prediction']);result=score_predictions(kind,y,arr['prediction'])
            if 'stage1_full_test' in row:assert row['stage1_full_test']==anchor
            row['stage1_full_test']=anchor;row[variant+'_stage2_test']=result;row[variant+'_raw_test']=score_predictions(kind,y,arr['raw_prediction']);row[variant+'_accepted']=calibrations[variant]['tasks'][t]['profile_acceptance'][calibrations[variant]['primary_profile']];row[variant+'_coverage']=float(arr['accepted'].mean());row[variant+'_profile']=calibrations[variant]['primary_profile']
            if not row[variant+'_accepted']:assert result==anchor
        results.append(row)
    assert len(results)==34
    atomic_json(root/'results.json',dict(tasks=results,protocol=PROTOCOL['evaluation_scope'],public_full_overrides=sorted(inputs.public_overrides)))
    atomic_json(root/'complete.json',dict(tasks=34,results_sha256=sha256(root/'results.json'),selection_sha256=sha256(root/'selection_locked.json'),test_predictions_sha256=sha256(root/'predictions_committed.json'),exact_fallback_checked=True,official_test_id_coverage_checked=True,time=time.time()))
    verify_source_lock()
    print('JARVIS_COMPLETE_ALL34',flush=True)
if __name__=='__main__':main()
