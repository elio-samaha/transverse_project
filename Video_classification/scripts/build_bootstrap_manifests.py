#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any

import av


NORMAL_ALIASES = {
    "training_normal_videos_anomaly",
    "normal",
    "normal_videos",
    "normalvideos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build binary/multiclass/multilabel manifests when only normal videos are available, "
            "and synthesize placeholder violent samples for pipeline testing."
        )
    )
    parser.add_argument("--normal-dir", required=True, help="Directory containing normal MP4 videos.")
    parser.add_argument(
        "--taxonomy-json",
        default="UCFCrime_Filtered_WithFilename.json",
        help="Taxonomy source JSON (raw UCFCrime JSON or extracted taxonomy JSON).",
    )
    parser.add_argument("--output-dir", default="manifests", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extensions", default=".mp4", help="Comma-separated video extensions.")
    parser.add_argument("--max-normal", type=int, default=60, help="Limit normal videos for quick experiments.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--normal-label", default="normal", help="Canonical normal class label.")
    parser.add_argument("--synthetic-train-per-class", type=int, default=2)
    parser.add_argument("--synthetic-val-per-class", type=int, default=1)
    parser.add_argument("--synthetic-test-per-class", type=int, default=1)
    parser.add_argument("--infer-duration", action="store_true", help="Probe video duration via PyAV.")
    return parser.parse_args()


def _split_camel(name: str) -> str:
    stage_1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    stage_2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", stage_1)
    return stage_2


def normalize_label(raw: str, normal_label: str) -> str:
    cleaned = _split_camel(raw).strip().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"__+", "_", cleaned).strip("_").lower()
    if cleaned in NORMAL_ALIASES:
        return normal_label
    return cleaned


def infer_raw_class(video_id: str, payload: dict[str, Any]) -> str:
    filename = payload.get("filename")
    if isinstance(filename, str) and filename.strip():
        normalized = filename.strip().replace("\\", "/")
        if "/" in normalized:
            return normalized.split("/", 1)[0]
        return normalized
    stem = str(video_id).split("_")[0]
    stem = re.sub(r"\d+$", "", stem)
    return stem


def load_taxonomy_classes(path: Path, normal_label: str) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        multiclass = payload.get("multiclass_classes")
        if isinstance(multiclass, list) and multiclass:
            normalized = [str(x) for x in multiclass]
            violent = [label for label in normalized if label != normal_label]
            return normalized, violent

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported taxonomy JSON schema in {path}")

    counter: Counter[str] = Counter()
    for video_id, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        raw = infer_raw_class(str(video_id), meta)
        counter[normalize_label(raw, normal_label)] += 1

    classes = sorted(counter.keys())
    if normal_label in classes:
        classes = [normal_label] + [label for label in classes if label != normal_label]
    else:
        classes = [normal_label] + classes
    violent = [label for label in classes if label != normal_label]
    return classes, violent


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("train/val/test ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n >= 3:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = max(n - n_train - n_val, 1)

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    while n_train + n_val + n_test < n:
        n_train += 1
    return n_train, n_val, n_test


def probe_duration(video_path: Path) -> float | None:
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return float(container.duration / av.time_base)
    except Exception:
        return None
    return None


def write_csv(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "video_path", "split", "label", "labels", "duration_sec"],
        )
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict[str, str]]) -> dict[str, Any]:
    per_label = defaultdict(lambda: defaultdict(int))
    for row in records:
        per_label[row["label"]][row["split"]] += 1
    return {
        label: dict(sorted(split_counts.items()))
        for label, split_counts in sorted(per_label.items())
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    normal_dir = Path(args.normal_dir).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal directory not found: {normal_dir}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy JSON not found: {taxonomy_path}")

    multiclass_classes, violent_classes = load_taxonomy_classes(
        taxonomy_path,
        normal_label=args.normal_label,
    )
    if not violent_classes:
        raise ValueError("No violent classes found in taxonomy JSON.")

    exts = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
    files = [path.resolve() for path in sorted(normal_dir.rglob("*")) if path.is_file() and path.suffix.lower() in exts]
    if args.max_normal is not None:
        files = files[: args.max_normal]
    if not files:
        raise ValueError(f"No videos found in {normal_dir} with extensions={exts}")

    random.shuffle(files)
    n_train, n_val, n_test = split_indices(
        len(files),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    split_paths = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val : n_train + n_val + n_test],
    }

    duration_cache: dict[str, str] = {}
    if args.infer_duration:
        for path in files:
            duration = probe_duration(path)
            duration_cache[str(path)] = f"{duration:.4f}" if duration is not None else ""

    base_rows: dict[str, list[dict[str, str]]] = {}
    for split_name, split_files in split_paths.items():
        rows: list[dict[str, str]] = []
        for path in split_files:
            row = {
                "id": path.stem,
                "video_path": str(path),
                "split": split_name,
                "label": args.normal_label,
                "labels": args.normal_label,
                "duration_sec": duration_cache.get(str(path), ""),
            }
            rows.append(row)
        base_rows[split_name] = rows

    synthetic_plan = {
        "train": max(0, args.synthetic_train_per_class),
        "val": max(0, args.synthetic_val_per_class),
        "test": max(0, args.synthetic_test_per_class),
    }

    binary_records: list[dict[str, str]] = []
    multiclass_records: list[dict[str, str]] = []
    multilabel_records: list[dict[str, str]] = []

    for split_name, rows in base_rows.items():
        binary_records.extend(rows)
        multiclass_records.extend(rows)
        multilabel_records.extend(rows)

        if not rows:
            continue
        source_cycle = cycle(rows)

        # Binary synthetic placeholders: violent.
        for idx in range(synthetic_plan[split_name]):
            src = next(source_cycle)
            binary_records.append(
                {
                    **src,
                    "id": f"{src['id']}__syn_violent_{idx}",
                    "label": "violent",
                    "labels": "violent",
                }
            )

        # Multiclass synthetic placeholders: one class per violent type.
        for cls in violent_classes:
            for idx in range(synthetic_plan[split_name]):
                src = next(source_cycle)
                multiclass_records.append(
                    {
                        **src,
                        "id": f"{src['id']}__syn_{cls}_{idx}",
                        "label": cls,
                        "labels": cls,
                    }
                )
                multilabel_records.append(
                    {
                        **src,
                        "id": f"{src['id']}__syn_ml_{cls}_{idx}",
                        "label": "violent",
                        "labels": f"violent|{cls}",
                    }
                )

    binary_path = output_dir / "bootstrap_binary_manifest.csv"
    multiclass_path = output_dir / "bootstrap_multiclass_manifest.csv"
    multilabel_path = output_dir / "bootstrap_multilabel_manifest.csv"

    write_csv(binary_records, binary_path)
    write_csv(multiclass_records, multiclass_path)
    write_csv(multilabel_records, multilabel_path)

    summary = {
        "normal_source_dir": str(normal_dir),
        "taxonomy_json": str(taxonomy_path),
        "normal_count": len(files),
        "multiclass_classes": multiclass_classes,
        "multilabel_classes": [args.normal_label, "violent"] + violent_classes,
        "binary_manifest": str(binary_path),
        "multiclass_manifest": str(multiclass_path),
        "multilabel_manifest": str(multilabel_path),
        "binary_summary": summarize(binary_records),
        "multiclass_summary": summarize(multiclass_records),
        "multilabel_summary": summarize(multilabel_records),
        "synthetic_rows_note": (
            "Synthetic rows reuse normal video files with fake labels for pipeline debugging only."
        ),
    }
    summary_path = output_dir / "bootstrap_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote: {binary_path}")
    print(f"Wrote: {multiclass_path}")
    print(f"Wrote: {multilabel_path}")
    print(f"Wrote: {summary_path}")
    print("Detected multiclass labels:")
    for label in multiclass_classes:
        print(f"  - {label}")


if __name__ == "__main__":
    main()
