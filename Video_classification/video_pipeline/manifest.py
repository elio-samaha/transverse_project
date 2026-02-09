from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _parse_labels_field(raw: str | None) -> list[str]:
    if raw is None:
        return []
    value = raw.strip()
    if not value:
        return []
    return [token.strip() for token in value.split("|") if token.strip()]


def _normalize_record(record: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    out = dict(record)
    video_path = str(out.get("video_path", "")).strip()
    if not video_path:
        raise ValueError("Manifest record is missing required field: video_path")

    path = Path(video_path)
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    out["video_path"] = str(path)

    out.setdefault("split", "train")

    if isinstance(out.get("labels"), str):
        out["labels"] = _parse_labels_field(out["labels"])
    elif out.get("labels") is None:
        out["labels"] = []

    if out.get("label") is None and out["labels"]:
        out["label"] = out["labels"][0]

    if out.get("id") is None:
        out["id"] = path.stem

    duration = out.get("duration_sec")
    if duration in ("", None):
        out["duration_sec"] = None
    else:
        out["duration_sec"] = float(duration)

    return out


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    suffix = manifest_path.suffix.lower()
    manifest_dir = manifest_path.parent

    if suffix == ".csv":
        records: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(_normalize_record(row, manifest_dir))
        return records

    if suffix == ".jsonl":
        records = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(_normalize_record(json.loads(stripped), manifest_dir))
        return records

    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = list(payload.values())
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported JSON manifest schema in {manifest_path}")
        return [_normalize_record(item, manifest_dir) for item in payload]

    raise ValueError(f"Unsupported manifest file type: {manifest_path}")


def split_manifest_by_name(
    records: list[dict[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    return [row for row in records if str(row.get("split", "")).lower() == split_name.lower()]


def infer_class_names(
    records: list[dict[str, Any]],
    task_type: str,
    configured_class_names: list[str] | None = None,
) -> list[str]:
    if configured_class_names:
        return configured_class_names

    labels: list[str] = []
    for row in records:
        if task_type == "multilabel":
            labels.extend(row.get("labels", []))
        else:
            label = row.get("label")
            if label is not None:
                labels.append(str(label))

    unique = sorted(set(labels))
    if task_type == "binary":
        # Allows a normal-only quick run while preserving binary output shape.
        if not unique:
            return ["normal", "violent"]
        if len(unique) == 1:
            only = unique[0]
            if only.lower() == "normal":
                return ["normal", "violent"]
            return ["normal", only]
        return unique[:2]

    if not unique:
        raise ValueError("Could not infer any class names from manifest.")
    return unique
