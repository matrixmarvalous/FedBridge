#!/usr/bin/env python3
# Codex-added 2026-08-14: record copied-versus-release source checksums.
# Codex-modified 2026-08-18: resolve public Bridge paths through an explicit
# historical rename table and distinguish renames from other bounded edits.
# Codex-modified 2026-08-19: remove two unreferenced standalone legacy modules.
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RENAME_TABLE = RELEASE_ROOT / "docs" / "SOURCE_RENAMES.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_source_renames(path: Path = RENAME_TABLE) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        mapping = {row["release_path"]: row["source_path"] for row in rows}
    if len(mapping) != 8:
        raise ValueError(f"expected eight explicit source renames in {path}")
    return mapping


def source_path(
    release_relative: Path, source_renames: dict[str, str]
) -> tuple[Path, bool]:
    renamed_source = source_renames.get(release_relative.as_posix())
    if renamed_source is not None:
        return Path(renamed_source), True
    if release_relative.parts[:3] == ("src", "tianshou", "tianshou"):
        return Path("src/tianshou") / Path(*release_relative.parts[2:]), False
    return release_relative, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=RELEASE_ROOT / "SOURCE_MANIFEST.csv"
    )
    args = parser.parse_args()
    candidates = sorted(
        [path for path in (RELEASE_ROOT / "src").rglob("*.py")]
        + [path for path in (RELEASE_ROOT / "examples" / "policy").glob("*.py")]
    )
    source_renames = load_source_renames()
    rows = []
    for release_file in candidates:
        relative = release_file.relative_to(RELEASE_ROOT)
        relative_source, was_renamed = source_path(relative, source_renames)
        original = args.source_root / relative_source
        if not original.is_file():
            raise FileNotFoundError(f"missing source file: {relative_source}")
        source_hash = digest(original)
        release_hash = digest(release_file)
        if was_renamed:
            copy_status = "renamed_for_bridge_release"
        elif source_hash == release_hash:
            copy_status = "byte_preserved"
        else:
            copy_status = "modified_for_release"
        rows.append(
            {
                "release_path": relative.as_posix(),
                "source_path": relative_source.as_posix(),
                # Codex-modified 2026-08-14: an original runner hash can act
                # as a guessing oracle for the deliberately removed, low-entropy
                # experiment seed text. Preserve it only for byte-identical files.
                "source_sha256": (
                    source_hash if copy_status == "byte_preserved" else "withheld"
                ),
                "release_sha256": release_hash,
                "copy_status": copy_status,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} source records to {args.output}")


if __name__ == "__main__":
    main()
