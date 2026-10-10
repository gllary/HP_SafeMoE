from __future__ import annotations

JARVIS_TASKS = (
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "optb88vdw_total_energy",
    "ehull",
    "mbj_bandgap",
)

ROLE_TO_MANIFEST_KEY = {
    "train": "official_train_positions",
    "val": "official_val_positions",
    "test": "official_test_positions",
}

LATENT_DIM = 128
OFFICIAL_SPLIT = 0

ARCHITECTURE_CANDIDATES = (
    "proposal_full",
)


def expert_id(source_task: str) -> str:
    if source_task not in JARVIS_TASKS:
        raise ValueError(f"Unknown JARVIS task: {source_task}")
    return f"cogn_{source_task}"
