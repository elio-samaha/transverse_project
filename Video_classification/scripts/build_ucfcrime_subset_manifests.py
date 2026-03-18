#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build binary, multiclass, and multilabel manifests from the "
            "UCFCrime *_AnomalyTrain split JSON files and a local dataset root."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory that contains the video files.")
    parser.add_argument("--train-json", required=True, help="Train split JSON path.")
    parser.add_argument("--val-json", required=True, help="Validation split JSON path.")
    parser.add_argument("--test-json", required=True, help="Test split JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory where manifest files will be written.")
    parser.add_argument(
        "--video-extensions",
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated list of video extensions to index.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any referenced key is missing or resolves to duplicate files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def normalize_video_extensions(raw_extensions: str) -> tuple[str, ...]:
    exts = []
    for value in raw_extensions.split(","):
        ext = value.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    if not exts:
        raise ValueError("At least one video extension is required.")
    return tuple(exts)


def to_snake_case(value: str) -> str:
    value = value.strip().replace("-", "_").replace(" ", "_")
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def infer_class_name(key: str) -> str:
    match = re.match(r"([A-Za-z]+)", key)
    if match is None:
        raise ValueError(f"Could not infer class name from key: {key}")
    return to_snake_case(match.group(1))


def index_videos(dataset_root: Path, video_extensions: tuple[str, ...]) -> dict[str, list[Path]]:
    stem_to_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in video_extensions:
            continue
        stem_to_paths[path.stem].append(path.resolve())
    return stem_to_paths


def resolve_video_path(
    key: str,
    stem_to_paths: dict[str, list[Path]],
) -> tuple[Path | None, list[str]]:
    candidates = stem_to_paths.get(key, [])
    if len(candidates) == 1:
        return candidates[0], []
    if not candidates:
        return None, []
    return candidates[0], [str(path) for path in candidates]


def make_binary_row(
    key: str,
    split_name: str,
    class_name: str,
    video_path: Path,
    duration: float | int | None,
) -> dict[str, str]:
    binary_label = "normal" if class_name == "normal" else "violent"
    duration_text = "" if duration in (None, "") else f"{float(duration):.4f}"
    return {
        "id": key,
        "video_path": str(video_path),
        "split": split_name,
        "label": binary_label,
        "labels": binary_label,
        "duration_sec": duration_text,
    }


def make_multiclass_row(
    key: str,
    split_name: str,
    class_name: str,
    video_path: Path,
    duration: float | int | None,
) -> dict[str, str]:
    duration_text = "" if duration in (None, "") else f"{float(duration):.4f}"
    return {
        "id": key,
        "video_path": str(video_path),
        "split": split_name,
        "label": class_name,
        "labels": class_name,
        "duration_sec": duration_text,
    }


def make_multilabel_row(
    key: str,
    split_name: str,
    class_name: str,
    video_path: Path,
    duration: float | int | None,
) -> dict[str, str]:
    if class_name == "normal":
        labels = "normal"
    else:
        labels = f"violent|{class_name}"
    duration_text = "" if duration in (None, "") else f"{float(duration):.4f}"
    return {
        "id": key,
        "video_path": str(video_path),
        "split": split_name,
        "label": class_name,
        "labels": labels,
        "duration_sec": duration_text,
    }


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "video_path", "split", "label", "labels", "duration_sec"],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split"] == split_name]
        summary[split_name] = dict(sorted(Counter(row["label"] for row in split_rows).items()))
    return summary


def build_taxonomy(multiclass_rows: list[dict[str, str]], multilabel_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    multiclass_classes = sorted({row["label"] for row in multiclass_rows})
    multilabel_classes = set()
    for row in multilabel_rows:
        labels = [token.strip() for token in row["labels"].split("|") if token.strip()]
        multilabel_classes.update(labels)

    ordered_multilabel_classes: list[str] = []
    for preferred in ("normal", "violent"):
        if preferred in multilabel_classes:
            ordered_multilabel_classes.append(preferred)
            multilabel_classes.remove(preferred)
    ordered_multilabel_classes.extend(sorted(multilabel_classes))

    return {
        "binary_classes": ["normal", "violent"],
        "multiclass_classes": multiclass_classes,
        "multilabel_classes": ordered_multilabel_classes,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = normalize_video_extensions(args.video_extensions)
    split_jsons = {
        "train": Path(args.train_json).expanduser().resolve(),
        "val": Path(args.val_json).expanduser().resolve(),
        "test": Path(args.test_json).expanduser().resolve(),
    }

    splits = {split_name: load_json(path) for split_name, path in split_jsons.items()}
    stem_to_paths = index_videos(dataset_root, video_extensions)
    if not stem_to_paths:
        raise ValueError(
            f"No videos found under {dataset_root} with extensions {', '.join(video_extensions)}"
        )

    binary_rows: list[dict[str, str]] = []
    multiclass_rows: list[dict[str, str]] = []
    multilabel_rows: list[dict[str, str]] = []
    missing_keys: dict[str, list[str]] = {split_name: [] for split_name in SPLIT_ORDER}
    duplicate_matches: dict[str, list[str]] = {}

    for split_name in SPLIT_ORDER:
        split_records = splits[split_name]
        for key, payload in split_records.items():
            class_name = infer_class_name(key)
            duration = payload.get("duration")
            video_path, duplicate_paths = resolve_video_path(key, stem_to_paths)
            if duplicate_paths:
                duplicate_matches[key] = duplicate_paths
            if video_path is None:
                missing_keys[split_name].append(key)
                continue

            binary_rows.append(make_binary_row(key, split_name, class_name, video_path, duration))
            multiclass_rows.append(make_multiclass_row(key, split_name, class_name, video_path, duration))
            multilabel_rows.append(make_multilabel_row(key, split_name, class_name, video_path, duration))

    total_missing = sum(len(values) for values in missing_keys.values())
    if args.strict and (total_missing > 0 or duplicate_matches):
        raise SystemExit(
            "Failed to build manifests cleanly. "
            f"missing={total_missing}, duplicates={len(duplicate_matches)}"
        )

    binary_manifest = output_dir / "ucfcrime_binary_manifest.csv"
    multiclass_manifest = output_dir / "ucfcrime_multiclass_manifest.csv"
    multilabel_manifest = output_dir / "ucfcrime_multilabel_manifest.csv"
    taxonomy_path = output_dir / "ucfcrime_taxonomy.json"
    summary_path = output_dir / "build_summary.json"
    missing_path = output_dir / "missing_keys.json"
    duplicate_path = output_dir / "duplicate_matches.json"

    write_manifest(binary_rows, binary_manifest)
    write_manifest(multiclass_rows, multiclass_manifest)
    write_manifest(multilabel_rows, multilabel_manifest)

    taxonomy = build_taxonomy(multiclass_rows, multilabel_rows)
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    missing_path.write_text(json.dumps(missing_keys, indent=2), encoding="utf-8")
    duplicate_path.write_text(json.dumps(duplicate_matches, indent=2), encoding="utf-8")

    summary = {
        "dataset_root": str(dataset_root),
        "video_extensions": list(video_extensions),
        "indexed_video_stems": len(stem_to_paths),
        "train_records_requested": len(splits["train"]),
        "val_records_requested": len(splits["val"]),
        "test_records_requested": len(splits["test"]),
        "binary_manifest": str(binary_manifest),
        "multiclass_manifest": str(multiclass_manifest),
        "multilabel_manifest": str(multilabel_manifest),
        "taxonomy_json": str(taxonomy_path),
        "binary_counts_by_split": summarize_counts(binary_rows),
        "multiclass_counts_by_split": summarize_counts(multiclass_rows),
        "multilabel_counts_by_split": summarize_counts(multilabel_rows),
        "missing_keys": missing_keys,
        "duplicate_matches": duplicate_matches,
        "taxonomy": taxonomy,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {binary_manifest}")
    print(f"Wrote {multiclass_manifest}")
    print(f"Wrote {multilabel_manifest}")
    print(f"Wrote {taxonomy_path}")
    print(f"Wrote {summary_path}")
    if total_missing:
        print(f"Missing keys: {total_missing}")
    if duplicate_matches:
        print(f"Duplicate matches: {len(duplicate_matches)}")


if __name__ == "__main__":
    main()
