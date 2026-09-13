#!/usr/bin/env python3
# Codex-added 2026-08-14: validate the external EasyRL4Rec assets used by the release runners.
"""Check the minimum MovieLens and DeepFM assets required by this release."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


SCENES: Sequence[str] = ("hetero5", "homo3")
SUBSETS: Sequence[str] = ("sub1", "sub2", "sub3")
CLUSTERS: Sequence[str] = ("A", "B", "C")

RAW_FILES: Sequence[str] = (
    "movielens-1m-train.csv",
    "movielens-1m-test.csv",
    "movies.dat",
    "users.dat",
    "rating_matrix.csv",
)
PROCESSED_FILES: Sequence[str] = (
    "distance_mat.pickle",
    "feature_domination.pickle",
    "item_popularity_add1.pickle",
    "item_similarity_add1.pickle",
)


def required_relative_paths(scenes: Sequence[str] = SCENES) -> List[Path]:
    """Return the release's deterministic, minimum external-asset inventory."""
    if not scenes or any(scene not in SCENES for scene in scenes):
        raise ValueError("select at least one supported scene")
    required: List[Path] = []
    movie_root = Path("data") / "MovieLens"

    for scene in scenes:
        for subset in SUBSETS:
            raw_dir = movie_root / f"{scene}_data_raw_{subset}"
            processed_dir = movie_root / f"{scene}_data_processed_{subset}"
            required.extend(raw_dir / name for name in RAW_FILES)
            required.extend(processed_dir / name for name in PROCESSED_FILES)

    model_root = Path("saved_models") / "MovieLensEnv-v0" / "DeepFM"
    for scene in scenes:
        for cluster in CLUSTERS:
            message = f"{scene}_pointneg{cluster}"
            required.append(model_root / "params" / f"[{message}]_params.pickle")
            required.append(model_root / "matsPre" / f"[{message}]_matPre.pickle")
            required.extend(
                model_root / "models" / f"[{message}]_model_M{index}.pt"
                for index in range(5)
            )
            required.append(
                model_root / "embeddings" / f"[{message}]_emb_user_val_M0.pt"
            )
            required.append(
                model_root / "embeddings" / f"[{message}]_emb_item_val_M0.pt"
            )

    return required


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    manifest_path: Path, asset_root: Path, relative_paths: Iterable[Path]
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "size_bytes", "sha256")
        )
        writer.writeheader()
        for relative_path in relative_paths:
            full_path = asset_root / relative_path
            writer.writerow(
                {
                    "path": relative_path.as_posix(),
                    "size_bytes": full_path.stat().st_size,
                    "sha256": sha256_file(full_path),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the external MovieLens/DeepFM assets required by the "
            "EasyRL4Rec experiments."
        )
    )
    parser.add_argument(
        "--asset-root",
        required=True,
        type=Path,
        help=(
            "Root containing data/MovieLens and "
            "saved_models/MovieLensEnv-v0/DeepFM."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        metavar="CSV",
        help="Optionally write relative paths, byte sizes, and SHA-256 digests.",
    )
    parser.add_argument("--scene", choices=SCENES, help="Check one scene; default: both.")
    parser.add_argument(
        "--verify-manifest", type=Path, metavar="CSV",
        help="Verify every selected asset against recorded byte sizes and SHA-256 digests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root: Path = args.asset_root
    required = required_relative_paths((args.scene,) if args.scene else SCENES)

    if not asset_root.is_dir():
        print("ERROR: the supplied asset root is not a directory.", file=sys.stderr)
        return 2

    missing: List[Path] = []
    invalid: List[Path] = []
    total_bytes = 0
    for relative_path in required:
        full_path = asset_root / relative_path
        if not full_path.exists():
            missing.append(relative_path)
        elif not full_path.is_file() or full_path.stat().st_size <= 0:
            invalid.append(relative_path)
        else:
            total_bytes += full_path.stat().st_size

    if missing or invalid:
        if missing:
            print("ERROR: missing required asset files:", file=sys.stderr)
            for path in missing:
                print(f"  - {path.as_posix()}", file=sys.stderr)
        if invalid:
            print("ERROR: empty or non-regular asset files:", file=sys.stderr)
            for path in invalid:
                print(f"  - {path.as_posix()}", file=sys.stderr)
        return 2

    if args.verify_manifest is not None:
        try:
            with args.verify_manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            records = {row["path"]: row for row in rows}
            if len(records) != len(rows):
                raise ValueError("duplicate manifest paths")
            for relative_path in required:
                record = records[relative_path.as_posix()]
                full_path = asset_root / relative_path
                if (int(record["size_bytes"]) != full_path.stat().st_size
                        or record["sha256"] != sha256_file(full_path)):
                    raise ValueError(f"asset checksum mismatch: {relative_path.as_posix()}")
        except (OSError, ValueError, TypeError, KeyError, csv.Error) as error:
            print(f"ERROR: manifest verification failed: {error}", file=sys.stderr)
            return 2
        print(f"SHA-256 verified for {len(required)} selected asset files.")

    if args.write_manifest is not None:
        try:
            write_manifest(args.write_manifest, asset_root, required)
        except OSError:
            print("ERROR: could not write the requested manifest.", file=sys.stderr)
            return 1
        print(f"Manifest written with {len(required)} relative-path rows.")

    print(
        f"OK: found {len(required)} required asset files "
        f"({total_bytes / 1_000_000_000:.2f} GB)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
