#!/usr/bin/env python3
"""Build or verify the SHA-256 release manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.csv"
EXCLUDED_NAMES = {"RELEASE_MANIFEST.csv", ".DS_Store"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content() -> str:
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.parts)
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=("relative_path", "size_bytes", "sha256"), lineterminator="\n")
    writer.writeheader()
    for path in paths:
        writer.writerow(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = content()
    if args.check:
        if MANIFEST.read_text(encoding="utf-8") != expected:
            raise SystemExit("release manifest differs from repository contents")
        row_count = expected.count("\n") - 1
        print(f"Release manifest verified: {row_count} files")
        return
    MANIFEST.write_text(expected, encoding="utf-8")
    print(MANIFEST)


if __name__ == "__main__":
    main()
