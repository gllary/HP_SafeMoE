from __future__ import annotations

import pytest

from hpsafe_sota.experts.jmp_l_bridge import (
    FAST32_PROTOCOL,
    LATENT_DIM,
    LATENT_SOURCE,
    LEARNING_RATE,
    OFFICIAL_COMMIT,
    SUPPORTED_TASKS,
    TASK_SETTINGS,
    _effective_task_settings,
)


def test_jmp_l_identity_and_representation_are_pinned() -> None:
    assert OFFICIAL_COMMIT == "937b14874381d9b80809582e323ef82f2d4e1291"
    assert LEARNING_RATE == 8.0e-5
    assert LATENT_DIM == 256
    assert LATENT_SOURCE == "gemnet_oc_final_atom_energy_embedding_before_task_head"


def test_combined_bridge_covers_the_final_jmp_tasks() -> None:
    final_tasks = {"jdft2d", "perovskites", "phonons", "mp_gap", "mp_e_form"}
    assert final_tasks.issubset(SUPPORTED_TASKS)
    assert TASK_SETTINGS["phonons"]["reduction"] == "max"
    assert TASK_SETTINGS["jdft2d"]["precision"] == "32-true"


def test_final_32_epoch_protocol_is_explicit_and_limited_to_two_mp_tasks() -> None:
    mp_gap = _effective_task_settings("mp_gap", {"finetune_protocol": FAST32_PROTOCOL})
    mp_e_form = _effective_task_settings(
        "mp_e_form", {"finetune_protocol": FAST32_PROTOCOL}
    )
    assert mp_gap["precision"] == "32-true"
    assert mp_e_form["precision"] == "32-true"
    with pytest.raises(ValueError, match="supports only"):
        _effective_task_settings("jdft2d", {"finetune_protocol": FAST32_PROTOCOL})


def test_unknown_finetune_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported JMP-L finetune_protocol"):
        _effective_task_settings("mp_gap", {"finetune_protocol": "unknown"})
