#!/usr/bin/env python3
"""Download final model weights into their checkpoint inventory paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/gllary/HP_SafeMoE/releases/download/final-checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path: Path, record: dict) -> bool:
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and sha256(path) == record["sha256"])


def download(record: dict) -> str:
    path = (ROOT / record["path"]).resolve()
    if not path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError(f"Invalid checkpoint path: {record['path']}")
    if matches(path, record):
        return "verified"
    asset = "__".join(Path(record["path"]).parts[1:])
    request = Request(f"{RELEASE}/{quote(asset, safe='')}",
                      headers={"User-Agent": "HP-SafeMoE-checkpoints"})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    handle.write(block)
            if not matches(partial, record):
                raise ValueError(f"Checkpoint size or SHA-256 mismatch: {record['path']}")
            partial.replace(path)
            return "downloaded"
        except (OSError, URLError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Checkpoint download did not complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("matbench", "jarvis"))
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--list", action="store_true", help="List selected files")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "checkpoints/required_checkpoints.json").read_text())
    records = []
    for package in inventory["required_packages"]:
        if args.benchmark and package["benchmark"].lower() != args.benchmark:
            continue
        if args.stage and package["stage"] != args.stage:
            continue
        records.extend(package["files"])
    total = sum(record["bytes"] for record in records)
    print(f"{len(records)} checkpoint files; {total / 1024 ** 3:.2f} GiB", flush=True)
    for index, record in enumerate(records, 1):
        if args.list:
            print(f"{record['path']} ({record['bytes']} bytes)")
        else:
            print(f"[{index}/{len(records)}] {record['path']}", flush=True)
            print(f"  {download(record)}", flush=True)


if __name__ == "__main__":
    main()
