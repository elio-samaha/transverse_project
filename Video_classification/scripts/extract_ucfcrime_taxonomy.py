#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


NORMAL_ALIASES = {
    "training_normal_videos_anomaly",
    "normal",
    "normal_videos",
    "normalvideos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract multiclass/multilabel taxonomy from UCFCrime filtered JSON."
    )
    parser.add_argument(
        "--input-json",
        default="UCFCrime_Filtered_WithFilename.json",
        help="Path to UCFCrime filtered JSON with filename field.",
    )
    parser.add_argument(
        "--output-json",
        default="manifests/ucfcrime_taxonomy.json",
        help="Path to output taxonomy JSON.",
    )
    parser.add_argument(
        "--normal-label",
        default="normal",
        help="Canonical label name used for normal videos.",
    )
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
        path = filename.strip().replace("\\", "/")
        if "/" in path:
            return path.split("/", 1)[0]
        return path

    stem = str(video_id).split("_")[0]
    stem = re.sub(r"\d+$", "", stem)
    return stem


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {input_path}, got {type(payload).__name__}")

    raw_counter: Counter[str] = Counter()
    normalized_counter: Counter[str] = Counter()
    raw_to_normalized: dict[str, str] = {}

    for video_id, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        raw_class = infer_raw_class(str(video_id), meta)
        normalized = normalize_label(raw_class, normal_label=args.normal_label)
        raw_counter[raw_class] += 1
        normalized_counter[normalized] += 1
        raw_to_normalized[raw_class] = normalized

    normal_label = args.normal_label
    violent_types = sorted(label for label in normalized_counter if label != normal_label)
    multiclass_classes = [normal_label] + violent_types
    multilabel_classes = [normal_label, "violent"] + violent_types

    taxonomy = {
        "source_json": str(input_path),
        "total_samples": int(sum(raw_counter.values())),
        "normal_label": normal_label,
        "raw_class_counts": dict(sorted(raw_counter.items())),
        "normalized_class_counts": dict(sorted(normalized_counter.items())),
        "raw_to_normalized_map": dict(sorted(raw_to_normalized.items())),
        "violent_type_classes": violent_types,
        "multiclass_classes": multiclass_classes,
        "multilabel_classes": multilabel_classes,
    }
    output_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Saved taxonomy to: {output_path}")
    print(f"Total samples: {taxonomy['total_samples']}")
    print(f"Detected classes ({len(multiclass_classes)} multiclass):")
    for label in multiclass_classes:
        count = normalized_counter.get(label, 0)
        print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()
