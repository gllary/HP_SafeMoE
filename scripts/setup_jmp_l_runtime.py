#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/JMP"
OFFICIAL_COMMIT = "937b14874381d9b80809582e323ef82f2d4e1291"
OFFICIAL_SOURCE_ARCHIVE_URL = (
    f"https://github.com/facebookresearch/JMP/archive/{OFFICIAL_COMMIT}.tar.gz"
)
OFFICIAL_CHECKPOINT_URL = "https://jmp-iclr-datasets.s3.amazonaws.com/jmp-l.pt"
OFFICIAL_CHECKPOINT_SIZE = 3_749_529_931
OFFICIAL_CHECKPOINT_ETAG = "81f27687ded066a05b58e75665d58675-447"
OFFICIAL_CHECKPOINT_ETAG_PART_SIZE = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _s3_multipart_etag(path: Path, part_size: int) -> str:
    part_digests: list[bytes] = []
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(part_size), b""):
            part_digests.append(hashlib.md5(block, usedforsecurity=False).digest())
    combined = hashlib.md5(b"".join(part_digests), usedforsecurity=False).hexdigest()
    return f"{combined}-{len(part_digests)}"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _head(source_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def setup_source(source_root: Path) -> None:
    if source_root.exists():
        if (source_root / ".git").is_dir():
            observed = _head(source_root)
        elif (source_root / "HPSAFE_SOURCE_COMMIT").is_file():
            observed = (source_root / "HPSAFE_SOURCE_COMMIT").read_text(
                encoding="utf-8"
            ).strip()
        else:
            raise RuntimeError(f"Refusing to replace unverified source directory: {source_root}")
        if observed != OFFICIAL_COMMIT:
            raise RuntimeError(
                f"Existing JMP checkout is {observed}; required commit is {OFFICIAL_COMMIT}"
            )
    else:
        source_root.parent.mkdir(parents=True, exist_ok=True)
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("curl is required to fetch the pinned official JMP source")
        archive = source_root.parent / f".JMP-{OFFICIAL_COMMIT}.tar.gz.part"
        subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--retry",
                "5",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(archive),
                OFFICIAL_SOURCE_ARCHIVE_URL,
            ],
            check=True,
        )
        temporary = Path(tempfile.mkdtemp(prefix=".JMP.extract.", dir=source_root.parent))
        try:
            temporary_resolved = temporary.resolve()
            with tarfile.open(archive, "r:gz") as handle:
                members = handle.getmembers()
                for member in members:
                    candidate = (temporary / member.name).resolve()
                    if not candidate.is_relative_to(temporary_resolved):
                        raise RuntimeError(f"Unsafe path in official JMP archive: {member.name}")
                    if member.issym() or member.islnk():
                        raise RuntimeError(
                            f"Unexpected link in official JMP archive: {member.name}"
                        )
                handle.extractall(temporary, members=members, filter="data")
            extracted = temporary / f"JMP-{OFFICIAL_COMMIT}"
            if not extracted.is_dir():
                raise RuntimeError(f"Official JMP archive has no expected root: {extracted}")
            os.replace(extracted, source_root)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        archive.unlink()
    scale_file = source_root / "src/jmp/models/gemnet/scale_files/large.pt"
    if not scale_file.is_file():
        raise FileNotFoundError(f"Pinned JMP checkout lacks the large scale file: {scale_file}")
    _atomic_text(source_root / "HPSAFE_SOURCE_COMMIT", OFFICIAL_COMMIT + "\n")
    _atomic_json(
        source_root / "HPSAFE_SOURCE_ORIGIN.json",
        {
            "schema_version": 1,
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "source_archive": OFFICIAL_SOURCE_ARCHIVE_URL,
            "license_note": "CC-BY-NC-4.0 majority; preserve upstream component notices",
        },
    )


def _checkpoint_is_current(checkpoint: Path) -> bool:
    checksum_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    if not checkpoint.is_file() or not checksum_path.is_file():
        return False
    if checkpoint.stat().st_size != OFFICIAL_CHECKPOINT_SIZE:
        return False
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    verification_path = checkpoint.with_suffix(checkpoint.suffix + ".verified.json")
    stat = checkpoint.stat()
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file()
        else {}
    )
    if (
        verification.get("sha256") == expected
        and verification.get("official_s3_etag") == OFFICIAL_CHECKPOINT_ETAG
        and int(verification.get("size_bytes", -1)) == stat.st_size
        and int(verification.get("mtime_ns", -1)) == stat.st_mtime_ns
    ):
        if verification.get("verification") != "full_file_sha256_and_official_s3_etag":
            _atomic_json(
                verification_path,
                {
                    **verification,
                    "verification": "full_file_sha256_and_official_s3_etag",
                },
            )
        return True

    # ZIP extraction changes mtime precision. Re-hash the relocated bundled checkpoint once,
    # then refresh the local stat receipt without contacting the public S3 endpoint.
    observed = _sha256(checkpoint)
    if observed != expected:
        raise RuntimeError(f"Bundled JMP-L checkpoint SHA-256 mismatch: {observed} != {expected}")
    observed_etag = _s3_multipart_etag(checkpoint, OFFICIAL_CHECKPOINT_ETAG_PART_SIZE)
    if observed_etag != OFFICIAL_CHECKPOINT_ETAG:
        raise RuntimeError(
            f"Bundled JMP-L official S3 ETag mismatch: {observed_etag} != "
            f"{OFFICIAL_CHECKPOINT_ETAG}"
        )
    _atomic_json(
        verification_path,
        {
            "schema_version": 1,
            "sha256": observed,
            "official_s3_etag": observed_etag,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "verification": "full_file_sha256_and_official_s3_etag",
        },
    )
    return True


def setup_checkpoint(checkpoint: Path, checkpoint_source: Path | None) -> str:
    if _checkpoint_is_current(checkpoint):
        checksum = checkpoint.with_suffix(checkpoint.suffix + ".sha256").read_text(
            encoding="utf-8"
        ).split()[0]
        origin_path = checkpoint.with_suffix(checkpoint.suffix + ".origin.json")
        origin = (
            json.loads(origin_path.read_text(encoding="utf-8"))
            if origin_path.is_file()
            else {}
        )
        _atomic_json(
            origin_path,
            {
                **origin,
                "schema_version": 1,
                "url": OFFICIAL_CHECKPOINT_URL,
                "expected_size_bytes": OFFICIAL_CHECKPOINT_SIZE,
                "expected_s3_etag": OFFICIAL_CHECKPOINT_ETAG,
                "sha256_after_download": checksum,
            },
        )
        return checksum
    if checkpoint.exists():
        raise RuntimeError(
            f"Checkpoint exists without a valid pinned receipt; move it aside before setup: {checkpoint}"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
    if checkpoint_source is not None:
        source = checkpoint_source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if partial.exists():
            raise RuntimeError(f"Partial checkpoint already exists: {partial}")
        shutil.copy2(source, partial)
    else:
        if not partial.is_file() or partial.stat().st_size != OFFICIAL_CHECKPOINT_SIZE:
            curl = shutil.which("curl")
            if curl is None:
                raise RuntimeError("curl is required for resumable JMP-L checkpoint download")
            subprocess.run(
                [
                    curl,
                    "--location",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    OFFICIAL_CHECKPOINT_URL,
                ],
                check=True,
            )
    if partial.stat().st_size != OFFICIAL_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"JMP-L checkpoint size is {partial.stat().st_size}; expected {OFFICIAL_CHECKPOINT_SIZE}. "
            f"The resumable partial was retained at {partial}."
        )
    observed_etag = _s3_multipart_etag(partial, OFFICIAL_CHECKPOINT_ETAG_PART_SIZE)
    if observed_etag != OFFICIAL_CHECKPOINT_ETAG:
        raise RuntimeError(
            f"JMP-L official S3 ETag mismatch: {observed_etag} != {OFFICIAL_CHECKPOINT_ETAG}"
        )
    checksum = _sha256(partial)
    os.replace(partial, checkpoint)
    stat = checkpoint.stat()
    _atomic_text(
        checkpoint.with_suffix(checkpoint.suffix + ".sha256"),
        f"{checksum}  {checkpoint.name}\n",
    )
    _atomic_json(
        checkpoint.with_suffix(checkpoint.suffix + ".verified.json"),
        {
            "schema_version": 1,
            "sha256": checksum,
            "official_s3_etag": observed_etag,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "verification": "full_file_sha256_and_official_s3_etag",
        },
    )
    _atomic_json(
        checkpoint.with_suffix(checkpoint.suffix + ".origin.json"),
        {
            "schema_version": 1,
            "url": OFFICIAL_CHECKPOINT_URL,
            "expected_size_bytes": OFFICIAL_CHECKPOINT_SIZE,
            "expected_s3_etag": OFFICIAL_CHECKPOINT_ETAG,
            "sha256_after_download": checksum,
            "download_completed_utc": datetime.now(UTC).isoformat(),
            "local_source": None if checkpoint_source is None else str(checkpoint_source.resolve()),
        },
    )
    return checksum


def install_editable(source_root: Path, project_root: Path) -> None:
    for path in (source_root, project_root):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                str(path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pin official JMP source and the public JMP-L pretrained checkpoint."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-source",
        type=Path,
        help="Copy an already downloaded official jmp-l.pt instead of downloading it.",
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Editable-install the pinned JMP checkout and this project into the current Python.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else project_root / "external/JMP"
    )
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else project_root / "checkpoints/jmp-l.pt"
    )
    setup_source(source_root)
    checksum = None
    if not args.skip_checkpoint:
        checksum = setup_checkpoint(checkpoint, args.checkpoint_source)
    if args.install:
        install_editable(source_root, project_root)
    print(
        json.dumps(
            {
                "status": "ready",
                "source_root": str(source_root),
                "source_commit": OFFICIAL_COMMIT,
                "checkpoint": None if args.skip_checkpoint else str(checkpoint),
                "checkpoint_sha256": checksum,
                "installed": bool(args.install),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
