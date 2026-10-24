"""Prediction-token normalization fitted on official training OOF rows."""
import hashlib
import json
import numpy as np
from .baseline_inputs import Inputs as BaselineInputs
from .common import ROOT, atomic_json


def scalar_stats(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size > 0 and np.isfinite(values).all()
    return float(values.mean()), float(max(values.std(), 1e-8))


class Inputs(BaselineInputs):
    def __init__(self):
        super().__init__()
        self.token_stats = {}
        # Compute the float32 prediction token used by the expert input adapter.
        oof, full = {}, {}
        for source in self.tasks:
            p = np.empty(len(self.ids), dtype=np.float32)
            for held in range(5):
                take = self.assignment == held
                p[take] = self.prediction[source][held][take]
            oof[source] = self.base_token(source, p)
            full[source] = self.base_token(source, np.asarray(self.prediction[source][-1], dtype=np.float32))
        distribution = {}
        for task in self.tasks:
            train_ids, val_ids = self.roles[task]['train'], self.roles[task]['val']
            rows = np.asarray([self.lookup[j] for j in train_ids])
            valrows = np.asarray([self.lookup[j] for j in val_ids])
            self.token_stats[task], distribution[task] = {}, {}
            for source in self.tasks:
                available = self.source_available(task, source, train_ids, self.assignment[rows])
                mean, scale = scalar_stats(oof[source][rows][available])
                self.token_stats[task][source] = dict(mean=mean, scale=scale, fit_rows=int(available.sum()))
                vmask = self.source_available(task, source, val_ids, np.full(len(val_ids), -1))
                vp = full[source][valrows][vmask].astype(np.float64)
                distribution[task][source] = dict(train_rows=int(available.sum()), val_rows=int(vmask.sum()),
                    before_train_mean=mean, before_train_std=float(oof[source][rows][available].astype(np.float64).std()),
                    before_val_mean=float(vp.mean()) if vp.size else None,
                    before_val_std=float(vp.std()) if vp.size else None,
                    after_val_mean=float(((vp-mean)/scale).mean()) if vp.size else None,
                    after_val_std=float((vp/scale).std()) if vp.size else None,
                    after_val_abs_max=float(np.max(np.abs((vp-mean)/scale))) if vp.size else None)
        payload = dict(policy='per-target/source prediction-token z-score; available official train OOF rows only',
            scale_floor=1e-8, classification_token='clipped logit',
            anchor_unchanged=True, hidden_unchanged=True, uncertainty_unchanged=True,
            fitted_on_val=False, fitted_on_test=False, test_labels_read=False,
            train_id_order_sha256={t:hashlib.sha256(json.dumps(self.roles[t]['train']).encode()).hexdigest() for t in self.tasks},
            statistics=self.token_stats)
        # Concurrent workers must agree exactly; do not overwrite a locked fit.
        destination = ROOT/'stage2/normalization.json'
        if destination.exists():
            assert json.loads(destination.read_text()) == payload
        else:
            atomic_json(destination, payload)
        atomic_json(ROOT/'stage2/input_distribution_audit.json', dict(tasks=distribution,
            scope='train and val features only; no test scores or test labels',
            diagnostic_only=True, automatic_hyperparameter_adjustment=False))

    def base_token(self, source, p):
        if self.records[self.tasks.index(source)]['metric'] == 'ACC':
            p = np.clip(p, 1e-5, 1-1e-5)
            return np.log(p/(1-p))
        mean, scale = self.stats[source]
        return (p-mean)/scale

    def source_available(self, task, source, jids, held):
        return np.ones(len(jids), dtype=bool)

    def batch(self, task, jids, role, device, dropout=0, rng=None, no_cross=False):
        batch = super().batch(task, jids, role, device, dropout, rng, no_cross)
        # Replace dictionary values, never mutate tensors in place: the target
        # anchor shares the original target token and MUST keep label units.
        for source in self.tasks:
            s = self.token_stats[task][source]
            batch['prediction_standard'][source] = (batch['prediction_standard'][source]-s['mean'])/s['scale']
        return batch
