from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hpsafe_sota.experts.target_firewall import TargetAccessFirewall


def test_target_firewall_allows_only_fit_rows() -> None:
    manifest = SimpleNamespace(targets=np.asarray([1.0, 2.0, 3.0, 4.0]))
    firewall = TargetAccessFirewall(
        manifest=manifest,
        allowed_fit_positions=np.asarray([0, 1]),
        prediction_positions=np.asarray([2, 3]),
    )
    assert np.array_equal(firewall.targets_for_fit(np.asarray([1, 0])), [2.0, 1.0])
    receipt = firewall.receipt()
    assert receipt["prediction_target_access_count"] == 0
    assert receipt["access_is_subset_of_fit"] is True


def test_target_firewall_rejects_prediction_labels() -> None:
    manifest = SimpleNamespace(targets=np.asarray([1.0, 2.0, 3.0]))
    firewall = TargetAccessFirewall(
        manifest=manifest,
        allowed_fit_positions=np.asarray([0, 1]),
        prediction_positions=np.asarray([2]),
    )
    with pytest.raises(PermissionError, match="outside the declared fitting rows"):
        firewall.targets_for_fit(np.asarray([2]))


def test_target_firewall_rejects_overlapping_roles() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        TargetAccessFirewall(
            manifest=SimpleNamespace(targets=np.asarray([1.0, 2.0])),
            allowed_fit_positions=np.asarray([0, 1]),
            prediction_positions=np.asarray([1]),
        )
