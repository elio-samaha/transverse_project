#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Colab launcher: writes runtime config in a persistent run directory and "
            "auto-resumes from checkpoint_last.pt when available."
        )
    )
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument("--run-dir", required=True, help="Persistent run directory (e.g., Google Drive).")
    parser.add_argument("--python-bin", default="python", help="Python executable.")
    parser.add_argument("--device", default=None, help="Optional device override (e.g. cuda:0).")
    parser.add_argument("--manifest", default=None, help="Optional manifest override.")
    parser.add_argument("--class-names-file", default=None, help="Optional class_names_file override.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the command.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def choose_resume_checkpoint(run_dir: Path) -> Path | None:
    last_path = run_dir / "checkpoint_last.pt"
    if last_path.exists():
        return last_path
    candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
    if candidates:
        return candidates[-1]
    return None


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["output_dir"] = str(run_dir)

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        cfg["manifest_path"] = str(manifest_path)
    if args.class_names_file:
        names_path = Path(args.class_names_file).expanduser().resolve()
        cfg["class_names_file"] = str(names_path)

    runtime_config_path = run_dir / "runtime_config.yaml"
    dump_yaml(cfg, runtime_config_path)

    resume_path = choose_resume_checkpoint(run_dir)
    command = [
        args.python_bin,
        "train_video_classifier.py",
        "--config",
        str(runtime_config_path),
    ]
    if resume_path is not None:
        command.extend(["--resume", str(resume_path)])
    if args.device:
        command.extend(["--device", args.device])

    command_str = " ".join(command)
    print(f"Runtime config: {runtime_config_path}")
    print(f"Run directory: {run_dir}")
    if resume_path is not None:
        print(f"Auto-resume checkpoint: {resume_path}")
    else:
        print("Auto-resume checkpoint: none (starting fresh)")
    print(f"Command: {command_str}")
    (run_dir / "launch_command.json").write_text(
        json.dumps({"command": command, "runtime_config": str(runtime_config_path)}, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
