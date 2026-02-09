#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare run metrics across multiple model output directories."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory produced by train_video_classifier.py (repeat per run).",
    )
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_metrics(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    test_metrics = run_dir / "metrics_test.json"
    if test_metrics.exists():
        return test_metrics, load_json(test_metrics)

    val_metrics = sorted(run_dir.glob("metrics_val_epoch_*.json"))
    if val_metrics:
        candidate = val_metrics[-1]
        return candidate, load_json(candidate)
    return None, None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw_dir in args.run_dir:
        run_dir = Path(raw_dir).expanduser().resolve()
        metrics_path, metrics = pick_metrics(run_dir)
        if metrics is None:
            print(f"[skip] No metrics found in {run_dir}")
            continue

        row = {
            "run_dir": str(run_dir),
            "metrics_file": str(metrics_path),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "micro_f1": float(metrics.get("micro_f1", 0.0)),
            "macro_precision": float(metrics.get("macro_precision", 0.0)),
            "macro_recall": float(metrics.get("macro_recall", 0.0)),
            "loss": float(metrics.get("loss", 0.0)),
            "total_support": int(metrics.get("total_support", 0)),
        }
        rows.append(row)

    if not rows:
        raise ValueError("No comparable runs found.")

    rows = sorted(rows, key=lambda item: item["macro_f1"], reverse=True)
    header = (
        "Rank  MacroF1  Accuracy  MicroF1  MacroP   MacroR   Loss     Support  RunDir"
    )
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:>4d}  {fmt(row['macro_f1']):>7}  {fmt(row['accuracy']):>8}  "
            f"{fmt(row['micro_f1']):>7}  {fmt(row['macro_precision']):>7}  "
            f"{fmt(row['macro_recall']):>7}  {fmt(row['loss']):>7}  "
            f"{row['total_support']:>7d}  {row['run_dir']}"
        )

    if args.output_csv:
        output_path = Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison CSV: {output_path}")


if __name__ == "__main__":
    main()
