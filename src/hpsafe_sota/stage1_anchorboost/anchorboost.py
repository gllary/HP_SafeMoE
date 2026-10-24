from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from hpsafe_sota.data.raw import load_task_definitions
from hpsafe_sota.experts.interface import ExpertBridge
from hpsafe_sota.experts.native_models import NativeDescriptorMLP
from hpsafe_sota.metrics import score_predictions
from hpsafe_sota.stage1_anchorboost.cache import AnchorBoostDescriptorCache


@dataclass
class PreparedAnchorData:
    cache: AnchorBoostDescriptorCache
    task_type: str
    train_positions: np.ndarray
    validation_positions: np.ndarray
    prediction_positions: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_prediction: np.ndarray
    x_train_tree: np.ndarray
    x_validation_tree: np.ndarray
    x_prediction_tree: np.ndarray
    x_train_neural: np.ndarray
    x_validation_neural: np.ndarray
    x_prediction_neural: np.ndarray
    imputer: SimpleImputer
    scaler: StandardScaler
    receipt: Path


@dataclass
class FittedAnchor:
    checkpoint: Path
    forest: Any
    xgb: Any
    mlp: torch.nn.Module
    target_mean: float
    target_scale: float
    blend_weights: np.ndarray
    candidate_names: list[str]
    hurdle: dict[str, Any] | None = None
    selected_iterations: int | None = None
    selected_mlp_epochs: int | None = None
    prediction: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    latents: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _config(project_root: Path, task_name: str) -> dict[str, Any]:
    payload = yaml.safe_load(
        (project_root / "configs/experiments/stage1_anchorboost_glass.yaml").read_text(
            encoding="utf-8"
        )
    )
    family = payload["model"]
    values = dict(family["defaults"])
    values.update(family.get("task_overrides", {}).get(task_name, {}))
    return values


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _metric(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(score_predictions(task_type, y_true, y_pred))


def _better(task_type: str, left: float, right: float) -> bool:
    return left > right if task_type == "classification" else left < right


def _blend_weights(task_type: str, y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n_models = matrix.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        prediction = matrix @ weights
        if task_type == "classification":
            clipped = np.clip(prediction, 1e-7, 1.0 - 1e-7)
            return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))
        return float(np.mean(np.abs(y - prediction)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda values: float(values.sum() - 1.0)},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    weights = result.x if result.success else initial
    weights = np.maximum(weights, 0.0)
    return weights / max(weights.sum(), 1e-12)


class AnchorBoostBridge(ExpertBridge):
    def describe(self) -> dict[str, Any]:
        return {
            "bridge": "anchorboost",
            "implementation": "first_party_enriched_descriptors_xgboost_extra_trees_residual_mlp",
            "official_model_claimed": False,
            "pretrained": False,
            "selection_metric": "raw_unit_mae_or_roc_auc",
            "latent": "512d_task_trained_residual_mlp",
        }

    def prepare(self) -> PreparedAnchorData:
        definition = load_task_definitions(self.request.project_path)[self.request.task_name]
        cache = AnchorBoostDescriptorCache(self.request.project_path, self.request.task_name)
        train = self.split.train_positions
        validation = self.split.validation_positions
        prediction = self.split.prediction_positions
        x_train_raw = cache.rows(train)
        x_validation_raw = cache.rows(validation)
        x_prediction_raw = cache.rows(prediction)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train_tree = imputer.fit_transform(x_train_raw).astype(np.float32)
        x_validation_tree = (
            imputer.transform(x_validation_raw).astype(np.float32)
            if len(validation)
            else np.empty((0, x_train_tree.shape[1]), dtype=np.float32)
        )
        x_prediction_tree = imputer.transform(x_prediction_raw).astype(np.float32)
        scaler = StandardScaler()
        x_train_neural = scaler.fit_transform(x_train_tree).astype(np.float32)
        x_validation_neural = (
            scaler.transform(x_validation_tree).astype(np.float32)
            if len(validation)
            else np.empty((0, x_train_tree.shape[1]), dtype=np.float32)
        )
        x_prediction_neural = scaler.transform(x_prediction_tree).astype(np.float32)
        receipt = self.request.output_path / "preprocessing_manifest.json"
        joblib.dump(
            {"imputer": imputer, "scaler": scaler},
            self.request.output_path / "feature_preprocessor.joblib",
        )
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "implementation": "anchorboost_enriched",
                    "task": self.request.task_name,
                    "descriptor_cache_receipt": str(cache.receipt_path),
                    "fit_positions": int(len(train)),
                    "validation_positions": int(len(validation)),
                    "prediction_positions": int(len(prediction)),
                    "feature_dimension": int(x_train_tree.shape[1]),
                    "imputer_fit_scope": "fit_partition_only",
                    "scaler_fit_scope": "fit_partition_only",
                    "target_used_for_features": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return PreparedAnchorData(
            cache=cache,
            task_type=definition.task_type,
            train_positions=train,
            validation_positions=validation,
            prediction_positions=prediction,
            y_train=self.manifest.targets[train].astype(np.float32),
            y_validation=self.manifest.targets[validation].astype(np.float32),
            y_prediction=self.manifest.targets[prediction].astype(np.float32),
            x_train_tree=x_train_tree,
            x_validation_tree=x_validation_tree,
            x_prediction_tree=x_prediction_tree,
            x_train_neural=x_train_neural,
            x_validation_neural=x_validation_neural,
            x_prediction_neural=x_prediction_neural,
            imputer=imputer,
            scaler=scaler,
            receipt=receipt,
        )

    def _fit_xgboost(self, prepared: PreparedAnchorData, values: dict[str, Any]) -> tuple[Any, int]:
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - deployment preflight catches this
            raise RuntimeError("AnchorBoost requires xgboost; create the hpsafe-anchorboost environment") from exc
        selected_iterations = int(values.get("selected_xgb_iterations", values["n_estimators"]))
        common = dict(
            n_estimators=selected_iterations,
            max_depth=int(values["max_depth"]),
            learning_rate=float(values["learning_rate"]),
            subsample=float(values["subsample"]),
            colsample_bytree=float(values["colsample_bytree"]),
            min_child_weight=float(values["min_child_weight"]),
            reg_lambda=float(values["reg_lambda"]),
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=self.request.seed,
            n_jobs=12,
        )
        if len(prepared.validation_positions):
            # XGBoost >=2.1 reads early stopping from the estimator constructor.
            # Production OOF/refit requests carry the selected iteration count and
            # have no validation set, so this branch is intentionally validation-only.
            common["early_stopping_rounds"] = int(values.get("xgb_patience", 80))
        if prepared.task_type == "classification":
            positives = max(float(prepared.y_train.sum()), 1.0)
            model: Any = xgb.XGBClassifier(
                **common,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=max((len(prepared.y_train) - positives) / positives, 1e-3),
            )
        else:
            model = xgb.XGBRegressor(**common, objective="reg:absoluteerror", eval_metric="mae")
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if len(prepared.validation_positions):
            fit_kwargs["eval_set"] = [(prepared.x_validation_tree, prepared.y_validation)]
        model.fit(prepared.x_train_tree, prepared.y_train, **fit_kwargs)
        best = getattr(model, "best_iteration", None)
        return model, int(best + 1 if best is not None else selected_iterations)

    def _fit_forest(self, prepared: PreparedAnchorData, values: dict[str, Any]) -> Any:
        common = dict(
            n_estimators=int(values["extra_trees_estimators"]),
            min_samples_leaf=int(values["extra_trees_min_samples_leaf"]),
            max_features=0.9,
            n_jobs=16,
            random_state=self.request.seed + 1,
        )
        model: Any
        if prepared.task_type == "classification":
            model = ExtraTreesClassifier(**common, class_weight="balanced")
        else:
            model = ExtraTreesRegressor(**common)
        model.fit(prepared.x_train_tree, prepared.y_train)
        return model

    def _fit_mlp(
        self, prepared: PreparedAnchorData, values: dict[str, Any]
    ) -> tuple[torch.nn.Module, float, float, int]:
        device = _device()
        model = NativeDescriptorMLP(
            prepared.x_train_neural.shape[1],
            hidden_dim=int(values["mlp_hidden_dim"]),
            latent_dim=512,
            layers=int(values["mlp_layers"]),
            dropout=float(values["mlp_dropout"]),
        ).to(device)
        classification = prepared.task_type == "classification"
        target_mean = 0.0 if classification else float(prepared.y_train.mean())
        target_scale = 1.0 if classification else max(float(prepared.y_train.std()), 1e-8)
        y_train = (
            prepared.y_train if classification else (prepared.y_train - target_mean) / target_scale
        ).astype(np.float32)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(values["learning_rate_mlp"]),
            weight_decay=float(values["weight_decay"]),
        )
        epochs = int(values.get("selected_mlp_epochs", values["mlp_epochs"]))
        patience = int(values["mlp_patience"])
        batch_size = int(values["mlp_batch_size"])
        x_train = torch.from_numpy(prepared.x_train_neural)
        y_tensor = torch.from_numpy(y_train)
        generator = torch.Generator().manual_seed(self.request.seed)
        positives = max(float(prepared.y_train.sum()), 1.0)
        pos_weight = torch.tensor(
            max((len(prepared.y_train) - positives) / positives, 1e-3), device=device
        )
        best_score = -math.inf if classification else math.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = epochs - 1
        stale = 0
        started = time.monotonic()
        training_log_path = self.request.output_path / "training_log.jsonl"
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(x_train), generator=generator)
            total = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                features = x_train[indices].to(device)
                target = y_tensor[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    raw, _ = model(features)
                    loss = (
                        F.binary_cross_entropy_with_logits(raw, target, pos_weight=pos_weight)
                        if classification
                        else 0.8 * F.smooth_l1_loss(raw, target) + 0.2 * F.l1_loss(raw, target)
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(indices)
            validation_score: float | None = None
            if len(prepared.validation_positions):
                prediction, _ = self._mlp_forward(
                    model,
                    prepared.x_validation_neural,
                    classification=classification,
                    target_mean=target_mean,
                    target_scale=target_scale,
                    batch_size=batch_size,
                    dropout=False,
                )
                validation_score = _metric(prepared.task_type, prepared.y_validation, prediction)
                if _better(prepared.task_type, validation_score, best_score):
                    best_score = validation_score
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
            record = {
                "event": "anchorboost_training_epoch",
                "stage": "anchorboost_mlp",
                "epoch": epoch + 1,
                "epochs": epochs,
                "percent": round(100.0 * (epoch + 1) / epochs, 2),
                "train_loss": total / max(len(x_train), 1),
                "validation_metric": validation_score,
                "metric": "roc_auc" if classification else "mae",
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            with training_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if len(prepared.validation_positions) and stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, target_mean, target_scale, best_epoch + 1

    @staticmethod
    def _mlp_forward(
        model: torch.nn.Module,
        x: np.ndarray,
        *,
        classification: bool,
        target_mean: float,
        target_scale: float,
        batch_size: int,
        dropout: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = next(model.parameters()).device
        model.train(dropout)
        predictions: list[np.ndarray] = []
        latents: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                raw, latent = model(torch.from_numpy(x[start : start + batch_size]).to(device))
                value = torch.sigmoid(raw) if classification else raw * target_scale + target_mean
                predictions.append(value.float().cpu().numpy())
                latents.append(latent.float().cpu().numpy())
        return np.concatenate(predictions), np.concatenate(latents)

    @staticmethod
    def _tree_prediction(model: Any, x: np.ndarray, classification: bool) -> np.ndarray:
        return model.predict_proba(x)[:, 1] if classification else model.predict(x)

    def _fit_hurdle(self, prepared: PreparedAnchorData, values: dict[str, Any]) -> dict[str, Any] | None:
        if self.request.task_name not in {"expt_gap", "mp_gap"}:
            return None
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Hurdle head requires xgboost") from exc
        nonmetal = prepared.y_train > 1e-10
        if nonmetal.sum() < 20 or (~nonmetal).sum() < 20:
            return None
        common = dict(
            n_estimators=max(300, int(values["n_estimators"] * 0.6)),
            max_depth=int(values["max_depth"]),
            learning_rate=float(values["learning_rate"]),
            subsample=float(values["subsample"]),
            colsample_bytree=float(values["colsample_bytree"]),
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=self.request.seed + 11,
            n_jobs=12,
        )
        metal_model = xgb.XGBClassifier(**common, objective="binary:logistic", eval_metric="auc")
        gap_model = xgb.XGBRegressor(**common, objective="reg:absoluteerror", eval_metric="mae")
        metal_model.fit(prepared.x_train_tree, (~nonmetal).astype(np.int8), verbose=False)
        gap_model.fit(prepared.x_train_tree[nonmetal], prepared.y_train[nonmetal], verbose=False)
        return {"metal": metal_model, "gap": gap_model}

    def _candidate_predictions(
        self,
        fitted: FittedAnchor,
        prepared: PreparedAnchorData,
        *,
        validation: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_tree = prepared.x_validation_tree if validation else prepared.x_prediction_tree
        x_neural = prepared.x_validation_neural if validation else prepared.x_prediction_neural
        classification = prepared.task_type == "classification"
        xgb_prediction = self._tree_prediction(fitted.xgb, x_tree, classification)
        forest_prediction = self._tree_prediction(fitted.forest, x_tree, classification)
        mlp_prediction, latent = self._mlp_forward(
            fitted.mlp,
            x_neural,
            classification=classification,
            target_mean=fitted.target_mean,
            target_scale=fitted.target_scale,
            batch_size=int(self.request.extra.get("mlp_batch_size", 2048)),
            dropout=False,
        )
        candidates = [xgb_prediction, forest_prediction, mlp_prediction]
        if fitted.hurdle is not None:
            p_metal = fitted.hurdle["metal"].predict_proba(x_tree)[:, 1]
            positive_gap = np.maximum(fitted.hurdle["gap"].predict(x_tree), 0.0)
            candidates.append((1.0 - p_metal) * positive_gap)
        return np.stack(candidates, axis=1).astype(np.float64), latent.astype(np.float32)

    def fit(self, prepared: PreparedAnchorData) -> FittedAnchor:
        _seed(self.request.seed)
        values = _config(self.request.project_path, self.request.task_name)
        values.update(self.request.extra)
        forest = self._fit_forest(prepared, values)
        print(json.dumps({"event": "anchorboost_stage_complete", "stage": "extra_trees"}), flush=True)
        xgb, xgb_iterations = self._fit_xgboost(prepared, values)
        print(json.dumps({"event": "anchorboost_stage_complete", "stage": "xgboost"}), flush=True)
        mlp, mean, scale, mlp_epochs = self._fit_mlp(prepared, values)
        hurdle = self._fit_hurdle(prepared, values)
        names = ["xgboost", "extra_trees", "residual_mlp"] + (["gap_hurdle"] if hurdle else [])
        fitted = FittedAnchor(
            checkpoint=self.request.output_path / "anchorboost.joblib",
            forest=forest,
            xgb=xgb,
            mlp=mlp,
            target_mean=mean,
            target_scale=scale,
            blend_weights=np.asarray(values.get("blend_weights", []), dtype=np.float64),
            candidate_names=names,
            hurdle=hurdle,
            selected_iterations=xgb_iterations,
            selected_mlp_epochs=mlp_epochs,
        )
        if len(prepared.validation_positions):
            matrix, _ = self._candidate_predictions(fitted, prepared, validation=True)
            fitted.blend_weights = _blend_weights(prepared.task_type, prepared.y_validation, matrix)
            candidate_scores = {
                name: _metric(prepared.task_type, prepared.y_validation, matrix[:, index])
                for index, name in enumerate(names)
            }
            blend_score = _metric(
                prepared.task_type, prepared.y_validation, matrix @ fitted.blend_weights
            )
        else:
            candidate_scores = {}
            blend_score = None
        if fitted.blend_weights.shape != (len(names),):
            fitted.blend_weights = np.full(len(names), 1.0 / len(names))
        fitted.summary = {
            "algorithm": "AnchorBoost",
            "candidate_names": names,
            "candidate_scores": candidate_scores,
            "blend_score": blend_score,
            "selected_parameters": {
                "blend_weights": fitted.blend_weights.tolist(),
                "selected_xgb_iterations": xgb_iterations,
                "selected_mlp_epochs": mlp_epochs,
            },
            "device": str(_device()),
        }
        checkpoint_payload = {
            "forest": forest,
            "xgb": xgb,
            "hurdle": hurdle,
            "mlp_state": {key: value.detach().cpu() for key, value in mlp.state_dict().items()},
            "mlp_architecture": {
                "input_dim": prepared.x_train_neural.shape[1],
                "hidden_dim": int(values["mlp_hidden_dim"]),
                "layers": int(values["mlp_layers"]),
                "dropout": float(values["mlp_dropout"]),
                "latent_dim": 512,
            },
            "target_mean": mean,
            "target_scale": scale,
            "blend_weights": fitted.blend_weights,
            "candidate_names": names,
            "summary": fitted.summary,
        }
        joblib.dump(checkpoint_payload, fitted.checkpoint, compress=3)
        (self.request.output_path / "native_training_summary.json").write_text(
            json.dumps(fitted.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return fitted

    def predict(
        self, fitted: FittedAnchor, prepared: PreparedAnchorData
    ) -> tuple[np.ndarray, np.ndarray | None]:
        matrix, latent = self._candidate_predictions(fitted, prepared, validation=False)
        prediction = matrix @ fitted.blend_weights
        if prepared.task_type == "classification":
            prediction = np.clip(prediction, 0.0, 1.0)
        fitted.prediction = prediction.astype(np.float64)
        fitted.uncertainty = matrix.std(axis=1).astype(np.float64)
        fitted.latents = latent
        return fitted.prediction, fitted.uncertainty

    def export_latent(
        self, fitted: FittedAnchor, prepared: PreparedAnchorData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if fitted.latents is None:
            raise RuntimeError("predict must run before export_latent")
        return fitted.latents, np.ones(len(fitted.latents), dtype=np.bool_)

    def checkpoint_path(self, fitted: FittedAnchor) -> Path | None:
        return fitted.checkpoint

    def preprocessing_manifest(self, prepared: PreparedAnchorData) -> Path:
        return prepared.receipt
