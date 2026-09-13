#!/usr/bin/env python3
# Codex-added 2026-08-14: build or verify one checksum inventory for the bundle.
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = RELEASE_ROOT / "MANIFEST.sha256"
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def included_files() -> list[Path]:
    """Return every distributable file except the self-referential manifest."""
    files: list[Path] = []
    for path in RELEASE_ROOT.rglob("*"):
        relative = path.relative_to(RELEASE_ROOT)
        if not path.is_file() or path == MANIFEST:
            continue
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(RELEASE_ROOT).as_posix())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_manifest() -> None:
    lines = [
        f"{digest(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}"
        for path in included_files()
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} checksums to {MANIFEST.name}")


def check_manifest() -> None:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", maxsplit=1)
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        expected[relative] = checksum
    actual_paths = {
        path.relative_to(RELEASE_ROOT).as_posix(): path for path in included_files()
    }
    if set(expected) != set(actual_paths):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise RuntimeError(f"manifest file-set mismatch; missing={missing}, extra={extra}")
    mismatches = [
        relative
        for relative, path in actual_paths.items()
        if digest(path) != expected[relative]
    ]
    if mismatches:
        raise RuntimeError(f"checksum mismatch: {mismatches}")
    print(f"OK: verified {len(expected)} release files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="verify MANIFEST.sha256 instead of rewriting it"
    )
    args = parser.parse_args()
    check_manifest() if args.check else write_manifest()


if __name__ == "__main__":
    main()
