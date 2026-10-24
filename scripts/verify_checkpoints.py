#!/usr/bin/env python3
"""Check final-model coverage, file integrity and optional Stage 2 loading."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict:
    inventory = json.loads((ROOT / 'checkpoints/required_checkpoints.json').read_text())
    assert inventory['status'] == 'available'
    assert inventory['public_weight_stages'] == [1, 2]
    packages = inventory['required_packages']
    assert len(packages) == 203 and len({p['id'] for p in packages}) == 203
    assert Counter((p['benchmark'], p['stage']) for p in packages) == {
        ('Matbench', 1): 65, ('JARVIS', 1): 34,
        ('Matbench', 2): 100, ('JARVIS', 2): 4,
    }
    assert inventory['required_public_stage2_members'] == 104
    assert inventory['stage2_variants'] == {'Full': 52, 'No-cross-expert': 52}
    assert Counter((p['benchmark'], p['variant']) for p in packages if p['stage'] == 2) == {
        ('Matbench', 'Full'): 50, ('Matbench', 'No-cross-expert'): 50,
        ('JARVIS', 'Full'): 2, ('JARVIS', 'No-cross-expert'): 2,
    }
    for artifact in inventory['shared_artifacts'].values():
        for relative in artifact if isinstance(artifact, list) else [artifact]:
            path = (ROOT / relative).resolve()
            assert path.is_relative_to(ROOT) and path.exists(), relative
    files = {}
    for package in packages:
        assert package['status'] == 'available' and package['role'] == 'final_prediction'
        assert len(package['source_checkpoint_sha256']) == 64
        if package['stage'] == 1:
            assert package['inference_config']
        for record in package['files']:
            path = (ROOT / record['path']).resolve()
            assert path.is_relative_to(ROOT / 'checkpoints') and path.is_file()
            assert record['path'] not in files
            assert path.stat().st_size == record['bytes'], record['path']
            assert sha256(path) == record['sha256'], record['path']
            files[record['path']] = record
    observed = {
        str(path.relative_to(ROOT))
        for path in (ROOT / 'checkpoints').rglob('*')
        if path.is_file() and path.suffix in {'.pt', '.pth', '.ckpt', '.joblib', '.pkl', '.h5'}
    }
    assert observed == set(files), 'Checkpoint files differ from the final-model inventory'
    assert len(files) == 208  # 203 model files and five fitted Glass preprocessors.
    # Source JARVIS hashes identify the original model files or provider archives.
    assets = json.loads((ROOT / 'results/JARVIS/experiment/stage1_assets.json').read_text())
    expected = {a['task']: a['checkpoint_sha256'] for a in assets['artifacts'] if a['fold'] == -1}
    for p in packages:
        if p['benchmark'] == 'JARVIS' and p['stage'] == 1:
            assert p['source_checkpoint_sha256'] == expected[p['task']]
    result = dict(model_packages=len(packages), weight_files=len(files),
                  stage1_packages=99, stage2_members=104,
                  bytes=sum(f['bytes'] for f in files.values()))
    print('Final checkpoint integrity verified:', json.dumps(result))
    return result


def load_stage2() -> None:
    import torch
    torch.set_num_threads(1)

    def module(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value

    matbench = module('checkpoint_matbench_model', 'src/hpsafe_sota/stage2/model.py')
    jarvis = module('checkpoint_jarvis_model', 'integrations/jarvis/src/jarvis_hpsafe/model.py')
    inventory = json.loads((ROOT / 'checkpoints/required_checkpoints.json').read_text())
    count = 0
    for package in inventory['required_packages']:
        if package['stage'] != 2:
            continue
        payload = torch.load(ROOT / package['files'][0]['path'], map_location='cpu', weights_only=True)
        if package['benchmark'] == 'Matbench':
            model = matbench.CrossExpertProposalModel(matbench.Stage2ProposalConfig(**payload['model_config']))
            state = payload['state_dict']
            assert set(payload['normalization']) == set(model.config.task_names)
        else:
            model = jarvis.CrossExpertProposalModel(jarvis.ProposalConfig(**payload['config']))
            state = payload['model']
            assert set(payload['target_normalization']) == set(model.config.task_names)
        assert model.config.use_cross_expert_input == (package['variant'] == 'Full')
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(t).all() for t in state.values())
        count += 1
    assert count == 104
    print('Full and No-cross-expert Stage 2 strict model loading verified:', count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-stage2', action='store_true')
    arguments = parser.parse_args()
    verify()
    if arguments.load_stage2:
        load_stage2()
