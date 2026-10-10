from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hpsafe_sota.stage2_common.mattervial_replay import (
    MatterVialReplayError,
    ReplayFeatureCache,
    configured_id_alias,
)


def test_replay_feature_cache_selects_requested_rows_and_columns() -> None:
    cache = ReplayFeatureCache(
        root=Path("/unit"),
        receipt={},
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        columns=("x", "y", "z"),
        values=np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    value = cache.member_matrix(("z", "x"), row_positions=np.asarray([3, 1]))
    assert np.array_equal(value, np.asarray([[11, 9], [5, 3]], dtype=np.float32))


def test_configured_id_alias_requires_a_total_bijection() -> None:
    alias = {
        "policy": "configured_regex_full_bijection",
        "source_regex": r"source-(?P<row>\d+)",
        "target_template": "target-{row}",
        "expected_mapping_count": 2,
    }
    assert configured_id_alias(["source-1", "source-2"], alias) == [
        "target-1",
        "target-2",
    ]
    with pytest.raises(MatterVialReplayError, match="row count"):
        configured_id_alias(["source-1"], alias)


def test_member_matrix_returns_none_for_missing_descriptor() -> None:
    cache = ReplayFeatureCache(
        root=Path("/unit"),
        receipt={},
        sample_ids=np.asarray(["a"]),
        columns=("x",),
        values=np.asarray([[1.0]], dtype=np.float32),
    )
    assert cache.member_matrix(("missing",)) is None
