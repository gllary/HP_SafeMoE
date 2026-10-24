"""Shared contracts for frozen Stage-1 exports consumed by Stage 2.

The export interface reuses the Stage-1 ``holdout_train``, ``holdout_val``, and
``outer_test`` indices without creating another cross-validation layer.
"""

from hpsafe_sota.stage2_common.protocol import (
    CANONICAL_TASKS,
    SPLIT_ROLE_TO_MANIFEST_ROLE,
    ExpertPool,
    load_expert_pool,
)

__all__ = [
    "CANONICAL_TASKS",
    "SPLIT_ROLE_TO_MANIFEST_ROLE",
    "ExpertPool",
    "load_expert_pool",
]
