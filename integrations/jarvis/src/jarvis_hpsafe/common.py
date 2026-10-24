"""Filesystem and material-fold contracts for JARVIS expert exports."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
CONFIG=PACKAGE.parents[1]/'configs'
ROOT=Path(os.environ.get('HPSAFE_JARVIS_OUTPUT','run_outputs/jarvis')).resolve()
SOURCE_ROOT=Path(os.environ.get('HPSAFE_JARVIS_INPUT','inputs/jarvis')).resolve()
PUBLIC_ROOT=Path(os.environ.get('HPSAFE_JARVIS_PUBLIC_FULL',str(SOURCE_ROOT/'public_full'))).resolve()
PROTOCOL=json.loads((CONFIG/'protocol.json').read_text())
PUBLIC_TASKS=tuple(PROTOCOL['public_alignn_full_tasks'])
CONTEXT={'scientific_policy':PROTOCOL['evaluation_scope']}
def atomic_json(path,value):
    path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def protocol_hash():return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
def fold(jid):return int.from_bytes(hashlib.blake2b(('jarvis34-oof-v1:'+jid).encode(),digest_size=8).digest(),'little')%5
def registry():return json.loads((CONFIG/'registry.json').read_text())
def manifest(task):return json.loads((SOURCE_ROOT/'data/tasks'/task/'manifest.json').read_text())
def training_labels(task,heldout):
    labels=json.loads((SOURCE_ROOT/'data/tasks'/task/'train.json').read_text());assert set(labels)==set(manifest(task)['role_ids']['train'])
    return labels if heldout<0 else {j:y for j,y in labels.items() if fold(j)!=heldout}
def output_dir(task,heldout):return SOURCE_ROOT/'artifacts'/task/('full' if heldout<0 else f'fold{heldout}')
def validate_expert_protocol(task,config):
    assert config['fold'] in [-1,0,1,2,3,4]
    assert set(config['trained_ids'])==set(training_labels(task,config['fold']))
    assert config['test_labels_read'] is False

def source_hashes():
    return {str(p.relative_to(PACKAGE)):sha256(p) for p in sorted(PACKAGE.rglob('*.py'))} | {'../configs/'+p.name:sha256(p) for p in sorted(CONFIG.glob('*.json'))}
def verify_source_lock():
    assert json.loads((ROOT/'status/source_lock.json').read_text())==source_hashes(),'Source/configuration differs from the preflight lock'
