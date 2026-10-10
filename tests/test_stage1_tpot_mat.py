from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hpsafe_sota.stage1_tpot.features import featurize_official_steel_compositions
from hpsafe_sota.stage1_tpot.official import (
    ELEMENTS,
    HIDDEN_DIM,
    OFFICIAL_COMMIT,
    OFFICIAL_PIPELINE_SHA256,
    verify_official_assets,
)
from hpsafe_sota.stage1_tpot.pipeline import (
    fit_pipeline,
    fresh_official_pipeline,
    replay_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/experiments/stage1_tpot_mat_steels.yaml"


def test_official_tpot_assets_are_hash_pinned_and_statically_inventoried() -> None:
    if not (PROJECT_ROOT / "vendors/tpot_mat_official").is_dir():
        pytest.skip("official TPOT assets are installed separately from this repository")
    receipt = verify_official_assets(PROJECT_ROOT)
    assert receipt["commit"] == OFFICIAL_COMMIT
    assert receipt["files"]["tpot_best_pipeline.pkl"] == OFFICIAL_PIPELINE_SHA256
    assert receipt["pipeline_template_usage"] == "topology_clone_and_outer_fold_refit"
    assert len(receipt["pickle_global_inventory"]) == 8


def test_steels_feature_order_matches_official_loader() -> None:
    assert ELEMENTS == (
        "Al", "C", "Co", "Cr", "Fe", "Mn", "Mo", "N", "Nb", "Ni", "Si", "Ti", "V", "W"
    )
    value = featurize_official_steel_compositions(
        ["Fe0.62C0.1Mn1.2Si0.3", "Fe0.3Al0.2W0.5"]
    )
    assert value.shape == (2, 14)
    assert value[0].tolist() == [
        0.0, 0.1, 0.0, 0.0, 0.62, 1.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0
    ]
    assert value[1].tolist() == [
        0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
    ]
    with pytest.raises(ValueError, match="outside the official TPOT-Mat schema"):
        featurize_official_steel_compositions(["Zr0.1"])


def test_tpot_graph_is_five_fixed_cpu_fits(tmp_path: Path) -> None:
    import importlib.util

    script = PROJECT_ROOT / "scripts/render_stage1_tpot_mat_steels.py"
    spec = importlib.util.spec_from_file_location("render_stage1_tpot_mat_steels", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = module.render(
        output_root=tmp_path / "orchestration",
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        config_path=CONFIG,
    )
    assert graph["schema_version"] == "tpot-mat-stage1-run"
    assert graph["graph_kind"].endswith("fe14")
    assert len(graph["jobs"]) == 13
    assert graph["fit_jobs"] == 5
    assert graph["gpu_jobs"] == 0
    assert graph["scheduler_profile"]["cpu_concurrency"] == 8
    assert set(graph["scheduler_profile"]["gpu_lanes"]) == {"0", "1", "2", "3"}
    fits = [job for job in graph["jobs"] if job["kind"] == "stage1_tpot_fixed_fold_fit"]
    assert len(fits) == 5
    assert {job["outer_fold"] for job in fits} == set(range(5))
    assert {job["seed"] for job in fits} == {18012019}
    assert all(job["gpu_lane"] is None for job in fits)
    assert graph["protocol"]["seed_scope"] == "all_outer_folds"
    assert graph["protocol"]["fit_scope"] == "outer_train"


def test_official_pipeline_refit_emits_genuine_103d_task_trained_state() -> None:
    pytest.importorskip("tpot")
    pipeline, updates = fresh_official_pipeline(PROJECT_ROOT, seed=20260901, threads=1)
    rng = np.random.default_rng(20260901)
    x = rng.uniform(0.0, 2.0, size=(80, len(ELEMENTS)))
    y = 100.0 + 5.0 * x[:, 1] - 3.0 * x[:, 4] + rng.normal(0.0, 0.2, size=80)
    pipeline, recipe = fit_pipeline(pipeline, x[:64], y[:64])
    replay = replay_pipeline(pipeline, recipe, x[64:])
    assert updates
    assert replay.prediction.shape == (16,)
    assert replay.hidden.shape == (16, HIDDEN_DIM)
    assert replay.component_predictions.shape == (16, 4)
    assert np.all(np.isfinite(replay.hidden))
    assert np.all(replay.uncertainty >= 0.0)
