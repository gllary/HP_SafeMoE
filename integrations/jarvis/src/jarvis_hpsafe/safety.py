from __future__ import annotations
import math
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import score_predictions

SAFETY_PROFILES: dict[str, dict[str, float | int]] = {
    "conservative": {
        "minimum_oof_gain_percent": 0.25,
        "bootstrap_lower_floor_percent": 0.00,
        "required_positive_partitions": 4,
        "maximum_partition_regret_percent": 0.50,
    },
    "balanced": {
        "minimum_oof_gain_percent": 0.10,
        "bootstrap_lower_floor_percent": -0.25,
        "required_positive_partitions": 3,
        "maximum_partition_regret_percent": 1.00,
    },
    "liberal": {
        "minimum_oof_gain_percent": 0.00,
        "bootstrap_lower_floor_percent": -1.00,
        "required_positive_partitions": 2,
        "maximum_partition_regret_percent": 3.00,
    },
}

def normalized_gain(task_type: str, anchor: float, model: float) -> float:
    if task_type == "classification":
        return 100.0 * (model - anchor) / max(1.0 - anchor, 1.0e-12)
    return 100.0 * (anchor - model) / max(abs(anchor), 1.0e-12)

def _ensemble_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("At least one proposal record is required")
    if any(not np.array_equal(records[0]["sample_ids"], record["sample_ids"]) for record in records[1:]):
        raise ValueError("Proposal ensemble sample IDs differ")
    anchor = np.asarray(records[0]["anchor_prediction"], dtype=np.float64)
    raw_members = np.stack([np.asarray(record["raw_prediction"], dtype=np.float64) for record in records])
    residual_members = raw_members - anchor[None, :]
    denominator = np.mean(np.abs(residual_members), axis=0)
    directional = np.divide(
        np.abs(np.mean(residual_members, axis=0)),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1.0e-12,
    )
    route_members = np.stack([record["route_index"] for record in records])
    route_consensus = np.max(np.stack([(route_members == route).mean(axis=0) for route in range(3)]), axis=0)
    return {
        "sample_ids": np.asarray(records[0]["sample_ids"]).astype(str),
        "anchor_prediction": anchor,
        "raw_prediction": raw_members.mean(axis=0),
        "member_raw_prediction": raw_members,
        "data_prediction": np.mean([record["data_prediction"] for record in records], axis=0),
        "physics_prediction": np.mean([record["physics_prediction"] for record in records], axis=0),
        "route_probabilities": np.mean([record["route_probabilities"] for record in records], axis=0).astype(
            np.float32
        ),
        "route_index": np.asarray(
            np.mean([record["route_probabilities"] for record in records], axis=0).argmax(axis=1),
            dtype=np.int8,
        ),
        "route_consensus": route_consensus.astype(np.float32),
        "directional_agreement": directional.astype(np.float32),
        "data_attention": np.mean([record["data_attention"] for record in records], axis=0),
        "physics_attention": np.mean([record["physics_attention"] for record in records], axis=0),
        "reliability": np.mean([record["reliability"] for record in records], axis=0),
    }

def _gate_features(values: dict[str, np.ndarray]) -> np.ndarray:
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    raw = np.asarray(values["raw_prediction"], dtype=np.float64)
    residual = raw - anchor
    members = np.asarray(values["member_raw_prediction"], dtype=np.float64)
    data = np.asarray(values["data_prediction"], dtype=np.float64)
    physics = np.asarray(values["physics_prediction"], dtype=np.float64)
    route = np.asarray(values["route_probabilities"], dtype=np.float64)
    data_attention = np.asarray(values["data_attention"], dtype=np.float64)
    physics_attention = np.asarray(values["physics_attention"], dtype=np.float64)
    reliability = np.asarray(values["reliability"], dtype=np.float64)

    def entropy(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1.0e-12, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=1)

    return np.column_stack(
        [
            anchor,
            raw,
            residual,
            np.abs(residual),
            members.std(axis=0),
            np.asarray(values["directional_agreement"]),
            np.asarray(values["route_consensus"]),
            data - physics,
            np.abs(data - physics),
            route,
            route.max(axis=1),
            entropy(route),
            data_attention.max(axis=1),
            physics_attention.max(axis=1),
            entropy(data_attention),
            entropy(physics_attention),
            reliability,
        ]
    ).astype(np.float64)

def _standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    scale = np.asarray(features, dtype=np.float64).std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return (features - mean) / scale, mean, scale

def _fit_benefit_gate(
    *, features: np.ndarray, labels: np.ndarray, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    maximum = int(config["benefit_gate_max_rows"])
    if len(label_array) > maximum:
        rng, pieces = np.random.default_rng(seed), []
        for label in (0.0, 1.0):
            candidates = np.flatnonzero(label_array == label)
            take = min(len(candidates), maximum // 2)
            pieces.append(rng.choice(candidates, size=take, replace=False))
        selected = np.concatenate(pieces)
        rng.shuffle(selected)
        feature_array, label_array = feature_array[selected], label_array[selected]
    x, mean, scale = _standardize_fit(feature_array)
    y = label_array
    if len(np.unique(y)) < 2:
        probability = float(np.mean(y))
        return {
            "kind": "constant",
            "constant_probability": probability,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "weight": [0.0] * x.shape[1],
            "bias": float(math.log((probability + 1.0e-4) / (1.0001 - probability))),
            "fit_rows": len(y),
        }
    torch.manual_seed(seed)
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32)
    layer = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=float(config["benefit_gate_learning_rate"]),
        weight_decay=float(config["benefit_gate_weight_decay"]),
    )
    positive = float(y.sum())
    negative = float(len(y) - positive)
    positive_weight = torch.tensor(max(negative / max(positive, 1.0), 1.0))
    for _ in range(int(config["benefit_gate_steps"])):
        optimizer.zero_grad(set_to_none=True)
        logits = layer(tensor_x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, tensor_y, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()
    return {
        "kind": "linear_logistic",
        "constant_probability": None,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weight": layer.weight.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
        "bias": float(layer.bias.detach().cpu()),
        "fit_rows": len(y),
    }

def _benefit_probability(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(features), float(model["constant_probability"]), dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    logits = ((np.asarray(features) - mean) / scale) @ weight + float(model["bias"])
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))

def _meta_partitions(y: np.ndarray, task_type: str, partitions: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(y), dtype=np.int8)
    if task_type == "classification":
        for label in (0, 1):
            selected = np.flatnonzero(np.asarray(y) == label)
            rng.shuffle(selected)
            assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
    else:
        selected = np.arange(len(y), dtype=np.int64)
        rng.shuffle(selected)
        assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
    return assignment

def _apply_parameters(
    *,
    anchor: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    alpha: float,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = np.asarray(anchor) + float(alpha) * (np.asarray(raw) - np.asarray(anchor))
    accepted = (
        np.ones(len(proposal), dtype=np.bool_)
        if threshold is None
        else np.asarray(probability) >= float(threshold)
    )
    accepted &= float(alpha) > 0.0
    return np.where(accepted, proposal, anchor), accepted

def _parameter_grid(
    task_type: str, config: dict[str, Any], *, sample_gate: bool
) -> list[tuple[float, float | None]]:
    scales = list(map(float, config["residual_scale_grid"]))
    if task_type == "classification" or not sample_gate:
        return [(alpha, None) for alpha in scales]
    thresholds = list(map(float, config["benefit_probability_threshold_grid"]))
    return [(0.0, None)] + [(alpha, threshold) for alpha in scales if alpha > 0.0 for threshold in thresholds]

def _select_parameters(
    *,
    y: np.ndarray,
    anchor: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    sample_gate: bool,
) -> dict[str, Any]:
    anchor_score = score_predictions(task_type, y, anchor)
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for alpha, threshold in _parameter_grid(task_type, config, sample_gate=sample_gate):
        prediction, accepted = _apply_parameters(
            anchor=anchor,
            raw=raw,
            probability=probability,
            alpha=alpha,
            threshold=threshold,
        )
        coverage = float(accepted.mean())
        if alpha > 0.0 and coverage < float(config["minimum_sample_coverage"]):
            continue
        gain = normalized_gain(task_type, anchor_score, score_predictions(task_type, y, prediction))
    # Validation meta-cross-fitted gain is primary. Exact ties prefer less correction and
        # broader coverage. This convention is identical for every task.
        rank = (gain, -alpha, coverage, -(threshold or 0.0))
        record = {
            "alpha": alpha,
            "threshold": threshold,
            "gain_percent": gain,
            "coverage": coverage,
        }
        if best is None or rank > best[0]:
            best = (rank, record)
    if best is None:
        raise RuntimeError("No valid OOF calibration parameters")
    return best[1]

def _bootstrap_gain(
    *,
    task_type: str,
    y: np.ndarray,
    anchor: np.ndarray,
    model: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    y, anchor, model = map(lambda value: np.asarray(value, dtype=np.float64), (y, anchor, model))
    point = normalized_gain(
        task_type,
        score_predictions(task_type, y, anchor),
        score_predictions(task_type, y, model),
    )
    rng, gains = np.random.default_rng(seed), []
    maximum = int(config["bootstrap_max_rows"])
    for _ in range(int(config["bootstrap_resamples"])):
        if task_type == "classification":
            pieces = []
            for label in (0, 1):
                candidates = np.flatnonzero(y == label)
                pieces.append(rng.choice(candidates, size=min(len(candidates), maximum // 2), replace=True))
            selected = np.concatenate(pieces)
        else:
            selected = rng.integers(0, len(y), size=min(len(y), maximum))
        gains.append(
            normalized_gain(
                task_type,
                score_predictions(task_type, y[selected], anchor[selected]),
                score_predictions(task_type, y[selected], model[selected]),
            )
        )
    values = np.asarray(gains, dtype=np.float64)
    confidence = float(config["bootstrap_confidence"])
    return {
        "point_gain_percent": point,
        "lower_gain_percent": float(np.quantile(values, 1.0 - confidence)),
        "upper_gain_percent": float(np.quantile(values, confidence)),
    }

def _profile_acceptance(
    *,
    alpha: float,
    certificate: dict[str, float],
    partition_gains: list[float],
) -> dict[str, bool]:
    positive = sum(gain > 0.0 for gain in partition_gains)
    return {
        name: bool(
            alpha > 0.0
            and certificate["point_gain_percent"] >= float(profile["minimum_oof_gain_percent"])
            and certificate["lower_gain_percent"] >= float(profile["bootstrap_lower_floor_percent"])
            and positive >= int(profile["required_positive_partitions"])
            and min(partition_gains) >= -float(profile["maximum_partition_regret_percent"])
        )
        for name, profile in SAFETY_PROFILES.items()
    }

def _calibrate_task(
    *,
    records: list[dict[str, np.ndarray]],
    y_true: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    values = _ensemble_records(records)
    features = _gate_features(values)
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    raw = np.asarray(values["raw_prediction"], dtype=np.float64)
    regression = task_type == "regression"
    local_benefit = (np.abs(anchor - y) > np.abs(raw - y)).astype(np.float32)
    assignment = _meta_partitions(y, task_type, int(config["meta_crossfit_partitions"]), seed=seed)
    meta_prediction = anchor.copy()
    meta_accepted = np.zeros(len(y), dtype=np.bool_)
    meta_probability = np.ones(len(y), dtype=np.float64)
    partition_records: list[dict[str, Any]] = []
    for held in range(int(config["meta_crossfit_partitions"])):
        tune, certify = assignment != held, assignment == held
        if regression:
            gate = _fit_benefit_gate(
                features=features[tune],
                labels=local_benefit[tune],
                seed=seed + held * 7919,
                config=config,
            )
            tune_probability = _benefit_probability(gate, features[tune])
            held_probability = _benefit_probability(gate, features[certify])
        else:
            tune_probability = np.ones(int(tune.sum()), dtype=np.float64)
            held_probability = np.ones(int(certify.sum()), dtype=np.float64)
        selected = _select_parameters(
            y=y[tune],
            anchor=anchor[tune],
            raw=raw[tune],
            probability=tune_probability,
            task_type=task_type,
            config=config,
            sample_gate=regression,
        )
        prediction, accepted = _apply_parameters(
            anchor=anchor[certify],
            raw=raw[certify],
            probability=held_probability,
            alpha=float(selected["alpha"]),
            threshold=selected["threshold"],
        )
        meta_prediction[certify] = prediction
        meta_accepted[certify] = accepted
        meta_probability[certify] = held_probability
        held_gain = normalized_gain(
            task_type,
            score_predictions(task_type, y[certify], anchor[certify]),
            score_predictions(task_type, y[certify], prediction),
        )
        partition_records.append(
            {
                **selected,
                "held_partition": held,
                "held_gain_percent": held_gain,
                "held_coverage": float(accepted.mean()),
            }
        )
    alpha = float(np.median([float(row["alpha"]) for row in partition_records]))
    selected_thresholds = [
        float(row["threshold"])
        for row in partition_records
        if row["threshold"] is not None and float(row["alpha"]) > 0.0
    ]
    threshold = float(np.median(selected_thresholds)) if selected_thresholds else None
    final_gate = (
        _fit_benefit_gate(features=features, labels=local_benefit, seed=seed + 99991, config=config)
        if regression
        else {
            "kind": "disabled_for_auc_task",
            "feature_mean": features.mean(axis=0).tolist(),
            "feature_scale": np.where(features.std(axis=0) > 1.0e-8, features.std(axis=0), 1.0).tolist(),
            "weight": [0.0] * features.shape[1],
            "bias": 0.0,
            "constant_probability": 1.0,
        }
    )
    partition_gains = [float(row["held_gain_percent"]) for row in partition_records]
    certificate = _bootstrap_gain(
        task_type=task_type,
        y=y,
        anchor=anchor,
        model=meta_prediction,
        config=config,
        seed=seed + 1777,
    )
    profiles = _profile_acceptance(alpha=alpha, certificate=certificate, partition_gains=partition_gains)
    return {
        "status": "true_inner_oof_meta_crossfit_benefit_calibration_complete",
        "task_type": task_type,
        "alpha": alpha,
        "benefit_probability_threshold": threshold,
        "sample_gate_enabled": regression,
        "benefit_gate": final_gate,
        "meta_crossfit_partition_records": partition_records,
        "meta_crossfit_partition_gains_percent": partition_gains,
        "meta_crossfit_positive_partitions": int(sum(gain > 0.0 for gain in partition_gains)),
        "meta_crossfit_coverage": float(meta_accepted.mean()),
        "benefit_label_positive_fraction": float(local_benefit.mean()),
        "meta_probability_mean": float(meta_probability.mean()),
        "certificate": certificate,
        "profile_acceptance": profiles,
        "fit_labels": "matching_official_outer_train_only",
        "outer_test_labels_accessed": False,
    }

def _apply_calibration(
    records: list[dict[str, np.ndarray]],
    *,
    calibration: dict[str, Any],
    profile: str,
    use_sample_gate: bool = True,
    use_task_certificate: bool = True,
) -> dict[str, np.ndarray]:
    values = _ensemble_records(records)
    features = _gate_features(values)
    task_type = str(calibration["task_type"])
    if task_type == "regression" and use_sample_gate:
        probability = _benefit_probability(calibration["benefit_gate"], features)
        threshold = calibration["benefit_probability_threshold"]
    else:
        probability = np.ones(len(features), dtype=np.float64)
        threshold = None
    prediction, accepted = _apply_parameters(
        anchor=values["anchor_prediction"],
        raw=values["raw_prediction"],
        probability=probability,
        alpha=float(calibration["alpha"]),
        threshold=threshold,
    )
    task_accepted = bool(calibration["profile_acceptance"][profile])
    if use_task_certificate and not task_accepted:
        prediction = np.asarray(values["anchor_prediction"], dtype=np.float64).copy()
        accepted = np.zeros(len(prediction), dtype=np.bool_)
    return {
        "sample_ids": np.asarray(values["sample_ids"]).astype(str),
        "anchor_prediction": np.asarray(values["anchor_prediction"], dtype=np.float64),
        "raw_prediction": np.asarray(values["raw_prediction"], dtype=np.float64),
        "prediction": np.asarray(prediction, dtype=np.float64),
        "accepted": np.asarray(accepted, dtype=np.bool_),
        "benefit_probability": np.asarray(probability, dtype=np.float32),
        "task_certificate_accepted": np.full(len(prediction), task_accepted, dtype=np.bool_),
        "residual_scale": np.full(len(prediction), float(calibration["alpha"]), dtype=np.float32),
        "route_index": np.asarray(values["route_index"], dtype=np.int8),
        "route_probabilities": np.asarray(values["route_probabilities"], dtype=np.float32),
        "directional_agreement": np.asarray(values["directional_agreement"], dtype=np.float32),
        "route_consensus": np.asarray(values["route_consensus"], dtype=np.float32),
        "data_attention": np.asarray(values["data_attention"], dtype=np.float32),
        "physics_attention": np.asarray(values["physics_attention"], dtype=np.float32),
    }
