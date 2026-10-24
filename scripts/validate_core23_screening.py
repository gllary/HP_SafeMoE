from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "data/Core-23/screening_decisions.csv"


def _sample_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def main() -> None:
    with DECISIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_counts = {
        "included_core": 23,
        "included_sensitivity_only": 60,
        "excluded_no_structure_match": 11,
        "excluded_no_589_3_nm_scalar": 7,
    }
    counts = Counter(row["status"] for row in rows)
    if len(rows) != 101 or counts != expected_counts:
        raise ValueError(f"Unexpected Core-23 screening counts: {dict(counts)}")

    included = [row for row in rows if row["status"].startswith("included_")]
    dataset_rows = sorted(int(row["dataset_row"]) for row in included)
    if dataset_rows != list(range(83)):
        raise ValueError("The 83 structure-matched scalar candidates are not indexed exactly once")
    if any(row["reason_codes"] for row in rows if row["status"] == "included_core"):
        raise ValueError("A Core-23 record carries an exclusion or sensitivity reason")
    if any(not row["reason_codes"] for row in rows if row["status"] != "included_core"):
        raise ValueError("A non-core screening record lacks its reason code")

    expected_core_ids = {
        f"rii-{int(row['dataset_row']):04d}"
        for row in rows
        if row["status"] == "included_core"
    }
    material_ids = _sample_ids(ROOT / "data/Core-23/materials.csv")
    label_ids = _sample_ids(ROOT / "data/Core-23/labels.csv")
    if material_ids != expected_core_ids or label_ids != expected_core_ids:
        raise ValueError("Core-23 final records do not match the screening decisions")

    print(
        "Core-23 screening decisions verified: 101 composition candidates, "
        "83 structure-matched scalar candidates, and 23 core records"
    )


if __name__ == "__main__":
    main()
