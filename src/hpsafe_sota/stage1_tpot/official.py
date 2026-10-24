from __future__ import annotations

import hashlib
import json
import pickletools
from pathlib import Path
from typing import Any

OFFICIAL_COMMIT = "936176db18ca4cd7b38cbd957c017a5bac770c6b"
OFFICIAL_DIRECTORY = "benchmarks/matbench_v0.1_TPOT"
OFFICIAL_PIPELINE_SHA256 = "21a3f2e07020ce2437310e23aa8d8051542ed785949cf38b0a4b7b08a795bec0"
OFFICIAL_FILES = {
    "Matbench_steel_TPOT.ipynb": "4c666e4852d6ae0d27b06b5e8d2957bcf8b8dd52c46ac885d82823fe3adfbf61",
    "info.json": "7bb7175b1de6fdaffd5310ede5b39f1e8f85d8705e449ea7250cbaa52c218ab2",
    "results.json.gz": "d7cce0528740474f099abcd007197c1a05b9ed5dd273727418bb5af122c72e11",
    "tpot_best_pipeline.pkl": OFFICIAL_PIPELINE_SHA256,
    "utils.py": "df221d8279222c3bc3e47f612a514aebc114838f51c68dd0e065a8fcd3dd1df4",
}
EXPERT_ID = "tpot_mat_steels_anchor"
PROVIDER = "tpot_mat_official_pipeline"
HIDDEN_DIM = 103
STACKING_COUNT = 3
XGBOOST_TREE_COUNT = 100
# Exact alphabetic order produced by the official ``LoadExisting.cleaning``.
# Its 13-item ``possible_elements`` list is only used to INSERT missing values;
# it does not remove Fe already present in every canonical steels composition.
# The fitted official LassoLarsCV therefore expects 14 raw input columns.
ELEMENTS = (
    "Al", "C", "Co", "Cr", "Fe", "Mn", "Mo", "N", "Nb", "Ni", "Si", "Ti", "V", "W"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_asset_root(project_root: Path) -> Path:
    return (
        project_root
        / "vendors"
        / "tpot_mat_official"
        / OFFICIAL_COMMIT
        / "matbench_v0.1_TPOT"
    ).resolve()


def _pickle_global_inventory(path: Path) -> list[str]:
    """Return a static opcode inventory of module/name strings.

    The exact SHA-256 identifies the asset, and the opcode inventory provides
    human-readable evidence for the declared pipeline topology.
    """

    interesting: list[str] = []
    tokens = {
        "sklearn.pipeline",
        "tpot.builtins.stacking_estimator",
        "sklearn.linear_model._least_angle",
        "tpot.builtins.one_hot_encoder",
        "sklearn.ensemble._forest",
        "tpot.builtins.zero_count",
        "sklearn.svm._classes",
        "xgboost.sklearn",
    }
    for opcode, argument, _ in pickletools.genops(path.read_bytes()):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            value = str(argument)
            if value in tokens:
                interesting.append(value)
    return sorted(set(interesting))


def verify_official_assets(project_root: Path) -> dict[str, Any]:
    root = official_asset_root(project_root)
    observed: dict[str, str] = {}
    for filename, expected in OFFICIAL_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"Official TPOT-Mat asset hash drifted: {filename}: {digest}")
        observed[filename] = digest
    info = json.loads((root / "info.json").read_text(encoding="utf-8"))
    if info.get("algorithm") != "TPOT-Mat":
        raise ValueError("Official TPOT-Mat info.json identity drifted")
    declared_requirements = set(info.get("requirements", {}).get("python", ()))
    required_requirements = {
        "numpy==1.23.5",
        "pandas==1.5.1",
        "tpot==0.11.7",
        "scikit-learn==1.2.2",
        "matbench",
        "joblib",
    }
    if not required_requirements.issubset(declared_requirements):
        raise ValueError("Official TPOT-Mat dependency receipt drifted")
    inventory = _pickle_global_inventory(root / "tpot_best_pipeline.pkl")
    required = {
        "sklearn.pipeline",
        "tpot.builtins.stacking_estimator",
        "sklearn.linear_model._least_angle",
        "tpot.builtins.one_hot_encoder",
        "sklearn.ensemble._forest",
        "tpot.builtins.zero_count",
        "sklearn.svm._classes",
        "xgboost.sklearn",
    }
    if set(inventory) != required:
        raise ValueError(f"Official TPOT-Mat pickle class inventory drifted: {inventory}")
    return {
        "status": "pass",
        "repository": "https://github.com/materialsproject/matbench",
        "commit": OFFICIAL_COMMIT,
        "directory": OFFICIAL_DIRECTORY,
        "asset_root": str(root),
        "files": observed,
        "pickle_global_inventory": inventory,
        "declared_python_requirements": sorted(declared_requirements),
        "pipeline_template_usage": "topology_clone_and_outer_fold_refit",
    }


def load_official_pipeline_template(project_root: Path) -> Any:
    """Load the hash-pinned official object only so sklearn can clone its recipe."""

    verify_official_assets(project_root)
    import joblib

    path = official_asset_root(project_root) / "tpot_best_pipeline.pkl"
    value = joblib.load(path)
    if type(value).__module__ != "sklearn.pipeline" or type(value).__name__ != "Pipeline":
        raise TypeError("Official TPOT-Mat asset is no longer an sklearn Pipeline")
    names = [str(name) for name, _ in value.steps]
    if names != [
        "stackingestimator-1",
        "onehotencoder",
        "stackingestimator-2",
        "zerocount",
        "stackingestimator-3",
        "xgbregressor",
    ]:
        raise ValueError(f"Official TPOT-Mat pipeline steps drifted: {names}")
    first = value.named_steps["stackingestimator-1"].estimator
    if int(getattr(first, "n_features_in_", -1)) != len(ELEMENTS):
        raise ValueError("Official TPOT-Mat fitted pipeline does not expect 14 raw features")
    return value
