from __future__ import annotations

from pathlib import Path


def asset_environment(project_root: Path) -> dict[str, str]:
    """Use one project-local model cache during setup and training."""

    root = (project_root / "model_cache").resolve()
    return {
        "HPSAFE_MODEL_CACHE_ROOT": str(root),
        "TABPFN_MODEL_CACHE_DIR": str(root / "tabpfn"),
        "TABPFN_NO_BROWSER": "1",
        "HF_HOME": str(root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface" / "hub"),
        "TORCH_HOME": str(root / "torch"),
        "XDG_CACHE_HOME": str(root / "xdg"),
        "KERAS_HOME": str(root / "keras"),
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "WANDB_DIR": str(root / "wandb"),
        "DGLBACKEND": "pytorch",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
    }


def ensure_asset_directories(environment: dict[str, str]) -> None:
    root = Path(environment["HPSAFE_MODEL_CACHE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    for variable in (
        "TABPFN_MODEL_CACHE_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "KERAS_HOME",
        "MPLCONFIGDIR",
        "WANDB_DIR",
    ):
        Path(environment[variable]).mkdir(parents=True, exist_ok=True)
