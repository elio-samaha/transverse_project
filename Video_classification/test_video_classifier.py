from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from torch.utils.data import DataLoader

from video_pipeline.dataset import DatasetConfig, VideoClassificationDataset
from video_pipeline.manifest import infer_class_names, load_manifest, split_manifest_by_name
from video_pipeline.metrics import format_metrics_table
from video_pipeline.models import ModelConfig, build_model
from video_pipeline.trainer import build_loss_fn, evaluate
from video_pipeline.utils import get_env_device, load_yaml, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate video sentiment classifier checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--split", default="test", help="Split name to evaluate (default: test)")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest override path (overrides config.manifest_path).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for metrics. Defaults next to checkpoint.",
    )
    return parser.parse_args()


def _resolve_configured_class_names(config: dict[str, Any], task_type: str) -> list[str] | None:
    class_names = config.get("class_names")
    if class_names:
        return class_names

    class_names_file = config.get("class_names_file")
    if not class_names_file:
        return None

    path = resolve_path(class_names_file)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported class_names_file schema in {path}")

    key = config.get("class_names_key")
    if not key:
        key = "multilabel_classes" if task_type == "multilabel" else "multiclass_classes"
    selected = payload.get(key)
    if not isinstance(selected, list):
        raise ValueError(f"class_names key '{key}' not found as a list in {path}")
    return [str(x) for x in selected]


def _build_dataset_and_loader(
    config: dict[str, Any],
    split_name: str,
    class_to_idx: dict[str, int],
    manifest_override: str | None,
) -> DataLoader:
    task_type = str(config.get("task_type", "binary")).lower()
    manifest_path = resolve_path(manifest_override if manifest_override else config["manifest_path"])
    records = load_manifest(manifest_path)
    split_records = split_manifest_by_name(records, split_name)
    if not split_records:
        raise ValueError(f"No records found for split '{split_name}'")

    data_cfg = config.get("data", {})
    ds_config = DatasetConfig(
        task_type=task_type,
        num_frames=int(data_cfg.get("num_frames", 16)),
        image_size=int(data_cfg.get("image_size", 112)),
        clip_duration_sec=(
            float(data_cfg["clip_duration_sec"]) if data_cfg.get("clip_duration_sec") is not None else None
        ),
        random_clip_for_train=False,
    )
    dataset = VideoClassificationDataset(
        records=split_records,
        class_to_idx=class_to_idx,
        split=split_name,
        config=ds_config,
    )
    train_cfg = config.get("training", {})
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None:
        raise ValueError("checkpoint path is required")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - for older torch.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    task_type = str(config.get("task_type", "binary")).lower()
    manifest_path = resolve_path(args.manifest if args.manifest else config["manifest_path"])
    records = load_manifest(manifest_path)
    configured_class_names = _resolve_configured_class_names(config, task_type=task_type)
    class_names = infer_class_names(
        records=records,
        task_type=task_type,
        configured_class_names=configured_class_names,
    )

    # Prefer class mapping from checkpoint when available.
    ckpt_mapping = checkpoint.get("class_to_idx")
    if ckpt_mapping:
        class_to_idx = {name: int(idx) for name, idx in ckpt_mapping.items()}
        class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    else:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model_cfg_raw = config.get("model", {})
    model_config = ModelConfig(
        name=str(model_cfg_raw.get("name", "r3d_18")).lower(),
        pretrained=False,
        dropout=float(model_cfg_raw.get("dropout", 0.2)),
        freeze_backbone=False,
        hf_model_id=model_cfg_raw.get("hf_model_id"),
        hf_trust_remote_code=bool(model_cfg_raw.get("hf_trust_remote_code", True)),
        mmaction_config_path=model_cfg_raw.get("mmaction_config_path"),
        mmaction_checkpoint_path=model_cfg_raw.get("mmaction_checkpoint_path"),
        mmaction_load_strict=bool(model_cfg_raw.get("mmaction_load_strict", False)),
    )
    model = build_model(model_config, num_outputs=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = _build_dataset_and_loader(
        config,
        split_name=args.split,
        class_to_idx=class_to_idx,
        manifest_override=args.manifest,
    )

    device = get_env_device(args.device)
    model.to(device)
    criterion = build_loss_fn(task_type)
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        task_type=task_type,
        class_names=class_names,
        multilabel_threshold=float(config.get("training", {}).get("multilabel_threshold", 0.5)),
    )

    print(format_metrics_table(metrics))
    output_path = (
        resolve_path(args.output)
        if args.output
        else checkpoint_path.parent / f"metrics_{args.split}_from_{checkpoint_path.stem}.json"
    )
    save_json(metrics, output_path)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
