"""Frozen Stage-1 exports using role-specific raw-official-split inputs."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,registry,fold,output_dir,manifest,atomic_json,sha256,validate_expert_protocol,SOURCE_ROOT,PUBLIC_TASKS,PUBLIC_ROOT


class Inputs:
    def __init__(self):
        self.records=registry()['tasks'];self.tasks=tuple(r['id'] for r in self.records)
        self.ids=np.load(SOURCE_ROOT/'data/universe_ids.npy').astype(str);self.lookup={j:i for i,j in enumerate(self.ids)}
        self.assignment=np.asarray([fold(j) for j in self.ids]);self.hidden={};self.prediction={};self.stats={};self.trained={};self.labels={};self.roles={};self.dims={};self.public_overrides=set(PUBLIC_TASKS)
        artifact_audit=[]
        for r in self.records:
            task=r['id'];m=manifest(task);assert 'role_ids' in m,'Refresh manifests before Stage 2'
            self.roles[task]=m['role_ids'];self.labels[task]={role:json.loads((SOURCE_ROOT/'data/tasks'/task/(role+'.json')).read_text()) for role in ['train','val']}
            y=np.asarray(list(self.labels[task]['train'].values()),dtype=np.float64)
            classification=r['metric']=='ACC';self.stats[task]=(0.,1.) if classification else (float(y.mean()),float(max(y.std(),1e-8)))
            self.trained[task]={};self.hidden[task]={};self.prediction[task]={}
            for held in [-1,0,1,2,3,4]:
                if held == -1 and task in self.public_overrides:
                    out=PUBLIC_ROOT/task;done=json.loads((out/'complete.json').read_text())
                    assert done['status']=='complete' and done['rows']==len(self.ids)
                    assert done['official_split_membership_and_order_exact'] is True
                    assert done['trained_or_finetuned'] is False
                    assert done['universe_ids_sha256']==sha256(SOURCE_ROOT/'data/universe_ids.npy')
                    assert done['prediction_sha256']==sha256(out/'prediction.npy')
                    assert done['hidden_sha256']==sha256(out/'hidden.npy')
                    assert all(not ids for ids in m['within_task_role_overlap'].values())
                    trained=set(m['role_ids']['train'])
                    config_hash=sha256(out/'complete.json');checkpoint_hash=done['checkpoint_archive_sha256']
                    provenance='official_public_ALIGNN_full_override'
                else:
                    out=output_dir(task,held);done=json.loads((out/'complete.json').read_text());cfg=json.loads((out/'config.json').read_text())
                    assert done['rows']==len(self.ids) and cfg['test_labels_read'] is False
                    validate_expert_protocol(task,cfg)
                    assert cfg['fold']==held
                    checkpoints=[out/name for name in ['model.pt','model.weights.h5','model.joblib'] if (out/name).is_file()]
                    assert len(checkpoints)==1 and sha256(checkpoints[0])==done['checkpoint_sha256']
                    trained=set(cfg['trained_ids'])
                    expected=set(m['role_ids']['train']) if held<0 else {j for j in m['role_ids']['train'] if fold(j)!=held}
                    assert trained==expected
                    if held>=0:assert all(fold(j)!=held for j in trained)
                    config_hash=sha256(out/'config.json');checkpoint_hash=done['checkpoint_sha256']
                    provenance='full_fit' if held<0 else 'out_of_fold'
                self.trained[task][held]=trained
                self.hidden[task][held]=np.load(out/'hidden.npy',mmap_mode='r')
                self.prediction[task][held]=np.load(out/'prediction.npy',mmap_mode='r')
                assert self.hidden[task][held].shape==(len(self.ids),done['hidden_dim'])
                assert self.prediction[task][held].shape==(len(self.ids),)
                assert np.isfinite(self.hidden[task][held]).all() and np.isfinite(self.prediction[task][held]).all()
                if classification:assert np.all((self.prediction[task][held]>=0)&(self.prediction[task][held]<=1))
                artifact_audit.append(dict(task=task,fold=held,source=provenance,config_sha256=config_hash,checkpoint_sha256=checkpoint_hash,prediction_sha256=sha256(out/'prediction.npy'),hidden_sha256=sha256(out/'hidden.npy')))
            self.dims[task]=self.hidden[task][-1].shape[1]
        role_overlap={t:manifest(t)['within_task_role_overlap'] for t in self.tasks}
        atomic_json(ROOT/'stage2/label_boundary_audit.json',dict(official_partition_overlap=role_overlap,test_labels_read=False,oof_folds=5,stage1_weights_updated=False))
        atomic_json(ROOT/'stage2/input_audit.json',dict(artifacts=artifact_audit,task_count=len(self.tasks),fits=len(artifact_audit),public_full_overrides=list(PUBLIC_TASKS),public_checkpoint_never_duplicated_into_oof_folds=True,stage1_oof=True,oof_rule='jarvis34-oof-v1',official_roles_unchanged=True,role_specific_prediction_policy='OOF for train; full for val/test',frozen_weights=True,test_labels_read=False))

    def anchor(self,task,jids,role):
        """Exact unstandardized anchor; never substitute full-fit train values."""
        assert role in ['train','val','test']
        rows=np.asarray([self.lookup[j] for j in jids]);held=self.assignment[rows] if role=='train' else np.full(len(rows),-1)
        result=np.empty(len(rows),dtype=np.float64)
        for f in np.unique(held):
            mask=held==f;result[mask]=self.prediction[task][int(f)][rows[mask]]
        return result

    def batch(self,task,jids,role,device,dropout=0,rng=None,no_cross=False):
        import torch
        assert role in ['train','val','test']
        rows=np.asarray([self.lookup[j] for j in jids]);held=self.assignment[rows] if role=='train' else np.full(len(rows),-1)
        hidden={};pred={};uncertainty={};available=np.ones((len(rows),len(self.tasks)),dtype=bool)
        for number,source in enumerate(self.tasks):
            h=np.empty((len(rows),self.dims[source]),dtype=np.float32);p=np.empty(len(rows),dtype=np.float32)
            for f in np.unique(held):
                mask=held==f;h[mask]=self.hidden[source][int(f)][rows[mask]];p[mask]=self.prediction[source][int(f)][rows[mask]]
            source_class=self.records[number]['metric']=='ACC'
            if source_class:p=np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
            else:
                mean,scale=self.stats[source];p=(p-mean)/scale
            hidden[source]=torch.as_tensor(h,device=device);pred[source]=torch.as_tensor(p,device=device);uncertainty[source]=torch.zeros(len(rows),device=device)
        index=self.tasks.index(task);available[:,index]=True
        if no_cross:
            available[:]=False;available[:,index]=True
        elif dropout:
            assert rng is not None
            available&=rng.random(available.shape)>=dropout;available[:,index]=True
        return {'task_index':torch.full((len(rows),),index,device=device,dtype=torch.long),'anchor_standard':pred[task],'hidden':hidden,'prediction_standard':pred,'uncertainty_standard':uncertainty,'available':torch.as_tensor(available,device=device)}

    def restore(self,task,p):
        if self.records[self.tasks.index(task)]['metric']=='ACC':return 1/(1+np.exp(-np.clip(p,-40,40)))
        mean,scale=self.stats[task];return p*scale+mean
