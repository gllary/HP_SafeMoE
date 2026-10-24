from __future__ import annotations

import json
from pathlib import Path

from hpsafe_sota.experts.jmp_l_bridge import (
    FAST32_PROTOCOL as H20_FP32_PROTOCOL,
    FAST32_TASK_SETTINGS as H20_FP32_TASK_SETTINGS,
)
from hpsafe_sota.experts.jmp_l_h20_fp32_jobs import (
    JMP_ENV_PYTHON,
    render_jmp_l_h20_fp32_job_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fp32_recipe_is_single_clean_path() -> None:
    assert H20_FP32_PROTOCOL == "jmp_l_fp32_32epoch"
    assert H20_FP32_TASK_SETTINGS == {
        "mp_gap": {
            "batch_size": 16,
            "precision": "32-true",
            "reduction": "mean",
            "num_workers": 8,
        },
        "mp_e_form": {
            "batch_size": 32,
            "precision": "32-true",
            "reduction": "mean",
            "num_workers": 0,
        },
    }


def test_fp32_graph_is_final_official_outer_fivefold(tmp_path: Path) -> None:
    output = tmp_path / "fp32"
    graph = render_jmp_l_h20_fp32_job_graph(
        PROJECT_ROOT,
        output,
        runtime_root=PROJECT_ROOT,
        cache_root=tmp_path / "cache" / "jmp_l",
    )
    assert graph["protocol"] == H20_FP32_PROTOCOL
    assert graph["official_outer_folds"] == list(range(5))
    assert graph["tasks"] == ["mp_gap", "mp_e_form"]
    assert graph["learning_rate"] == 8.0e-5
    assert graph["maximum_epochs"] == 32
    assert graph["initialization"] == "official_jmp_l_pretrained_checkpoint"
    folds = [job for job in graph["jobs"] if job["kind"] == "jmp_l_official_finetune"]
    assert len(folds) == 10
    assert {job["gpu_lane"] for job in folds[:8]} == set(range(8))
    for job in graph["jobs"]:
        assert job["command"][0] == JMP_ENV_PYTHON
        assert "conda" not in job["command"]
    for job in folds:
        request = json.loads(Path(job["request_path"]).read_text(encoding="utf-8"))
        settings = H20_FP32_TASK_SETTINGS[job["task_name"]]
        assert request["phase"] == "official_finetune"
        assert request["seed"] == 42
        assert request["inner_fold"] is None
        assert request["extra"]["finetune_protocol"] == H20_FP32_PROTOCOL
        assert request["extra"]["precision"] == "32-true"
        assert request["extra"]["batch_size"] == settings["batch_size"]
        assert request["extra"]["num_workers"] == settings["num_workers"]
        assert request["extra"]["latent_export"]["dimension"] == 256
