#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV manifest for binary/multiclass video classification."
    )
    parser.add_argument(
        "--class-dir",
        action="append",
        required=True,
        help="Class mapping in the form label=/path/to/videos. Repeat per class.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--extensions", default=".mp4", help="Comma-separated extensions.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--infer-duration", action="store_true", help="Probe video duration with PyAV.")
    parser.add_argument("--relative-to-output", action="store_true", help="Write paths relative to output file.")
    return parser.parse_args()


def parse_class_dirs(raw_items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --class-dir value: {item}. Expected label=/path")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not label:
            raise ValueError(f"Invalid class label in --class-dir: {item}")
        if not path.exists():
            raise FileNotFoundError(f"Class directory not found: {path}")
        mapping[label] = path
    return mapping


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

    train_count = int(round(n * train_ratio))
    val_count = int(round(n * val_ratio))
    test_count = n - train_count - val_count

    # Keep splits non-empty when possible.
    if n >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = max(1, n - train_count - val_count)

    while train_count + val_count + test_count > n:
        if train_count >= val_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < n:
        train_count += 1

    return train_count, val_count, test_count


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    class_dirs = parse_class_dirs(args.class_dir)
    exts = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list[Path]] = defaultdict(list)
    for label, class_dir in class_dirs.items():
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in exts:
                by_class[label].append(path.resolve())

        if args.max_per_class is not None:
            by_class[label] = by_class[label][: args.max_per_class]

    records: list[dict[str, str]] = []
    for label, files in by_class.items():
        if not files:
            continue
        random.shuffle(files)
        n_train, n_val, n_test = split_indices(
            len(files),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val : n_train + n_val + n_test],
        }
        for split_name, split_paths in split_files.items():
            for video_path in split_paths:
                write_path = (
                    str(video_path.relative_to(output_path.parent))
                    if args.relative_to_output
                    else str(video_path)
                )
                duration = probe_duration(video_path) if args.infer_duration else None
                records.append(
                    {
                        "id": video_path.stem,
                        "video_path": write_path,
                        "split": split_name,
                        "label": label,
                        "labels": label,
                        "duration_sec": f"{duration:.4f}" if duration is not None else "",
                    }
                )

    if not records:
        raise ValueError("No videos found with the specified --class-dir and --extensions.")

    random.shuffle(records)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "video_path", "split", "label", "labels", "duration_sec"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records to {output_path}")
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        summary[row["label"]][row["split"]] += 1
    for label, split_counts in sorted(summary.items()):
        counts_text = ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items()))
        print(f"  {label}: {counts_text}")


if __name__ == "__main__":
    main()
