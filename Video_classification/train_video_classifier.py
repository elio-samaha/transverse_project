from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from video_pipeline.dataset import DatasetConfig, VideoClassificationDataset
from video_pipeline.manifest import infer_class_names, load_manifest, split_manifest_by_name
from video_pipeline.metrics import format_metrics_table
from video_pipeline.models import ModelConfig, build_model
from video_pipeline.trainer import TrainConfig, fit
from video_pipeline.utils import get_env_device, load_yaml, resolve_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train video sentiment classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from (overrides config.checkpoint.resume_from).",
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest override path (overrides config.manifest_path).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional checkpoint output override (overrides config.checkpoint.output_dir).",
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


def _build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(cfg.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(cfg.get("t_max", epochs))
        eta_min = float(cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if config_path is None:
        raise ValueError("Config path is required.")
    config = load_yaml(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    task_type = str(config.get("task_type", "binary")).lower()
    manifest_path = resolve_path(args.manifest if args.manifest else config["manifest_path"])
    if manifest_path is None:
        raise ValueError("manifest_path is required in config.")
    records = load_manifest(manifest_path)
    configured_class_names = _resolve_configured_class_names(config, task_type=task_type)
    class_names = infer_class_names(
        records=records,
        task_type=task_type,
        configured_class_names=configured_class_names,
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Classes ({len(class_names)}): {class_names}")

    splits_cfg = config.get("splits", {})
    train_split_name = str(splits_cfg.get("train", "train"))
    val_split_name = str(splits_cfg.get("val", "val"))
    test_split_name = str(splits_cfg.get("test", "test"))

    train_records = split_manifest_by_name(records, train_split_name)
    val_records = split_manifest_by_name(records, val_split_name)
    test_records = split_manifest_by_name(records, test_split_name)
    if not train_records:
        raise ValueError(f"No records found for train split '{train_split_name}'")
    if not val_records:
        raise ValueError(f"No records found for val split '{val_split_name}'")

    data_cfg = config.get("data", {})
    ds_config = DatasetConfig(
        task_type=task_type,
        num_frames=int(data_cfg.get("num_frames", 16)),
        image_size=int(data_cfg.get("image_size", 112)),
        clip_duration_sec=(
            float(data_cfg["clip_duration_sec"]) if data_cfg.get("clip_duration_sec") is not None else None
        ),
        random_clip_for_train=bool(data_cfg.get("random_clip_for_train", True)),
    )

    train_dataset = VideoClassificationDataset(
        records=train_records,
        class_to_idx=class_to_idx,
        split="train",
        config=ds_config,
    )
    val_dataset = VideoClassificationDataset(
        records=val_records,
        class_to_idx=class_to_idx,
        split="val",
        config=ds_config,
    )
    test_dataset = (
        VideoClassificationDataset(
            records=test_records,
            class_to_idx=class_to_idx,
            split="test",
            config=ds_config,
        )
        if test_records
        else None
    )

    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(train_cfg.get("num_workers", 2))
    pin_memory = bool(train_cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if test_dataset is not None
        else None
    )

    model_cfg_raw = config.get("model", {})
    model_config = ModelConfig(
        name=str(model_cfg_raw.get("name", "r3d_18")).lower(),
        pretrained=bool(model_cfg_raw.get("pretrained", True)),
        dropout=float(model_cfg_raw.get("dropout", 0.2)),
        freeze_backbone=bool(model_cfg_raw.get("freeze_backbone", False)),
        hf_model_id=model_cfg_raw.get("hf_model_id"),
        hf_trust_remote_code=bool(model_cfg_raw.get("hf_trust_remote_code", True)),
        mmaction_config_path=model_cfg_raw.get("mmaction_config_path"),
        mmaction_checkpoint_path=model_cfg_raw.get("mmaction_checkpoint_path"),
        mmaction_load_strict=bool(model_cfg_raw.get("mmaction_load_strict", False)),
    )

    model = build_model(model_config, num_outputs=len(class_names))
    optimizer = _build_optimizer(model, config.get("optimizer", {}))
    scheduler = _build_scheduler(optimizer, config.get("scheduler", {}), epochs=int(train_cfg.get("epochs", 5)))

    device = get_env_device(args.device)
    train_config = TrainConfig(
        task_type=task_type,
        epochs=int(train_cfg.get("epochs", 5)),
        amp=bool(train_cfg.get("amp", True)) and device.type == "cuda",
        grad_clip_norm=(
            float(train_cfg["grad_clip_norm"]) if train_cfg.get("grad_clip_norm") is not None else None
        ),
        log_every=int(train_cfg.get("log_every", 10)),
        multilabel_threshold=float(train_cfg.get("multilabel_threshold", 0.5)),
    )

    ckpt_cfg = config.get("checkpoint", {})
    output_dir = resolve_path(
        args.output_dir if args.output_dir else ckpt_cfg.get("output_dir", "runs/video_classifier")
    )
    if output_dir is None:
        raise ValueError("Invalid checkpoint.output_dir")
    resume_path = args.resume if args.resume else ckpt_cfg.get("resume_from")

    print(
        f"Starting training on device={device} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset) if test_dataset else 0}"
    )
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_config=train_config,
        class_names=class_names,
        output_dir=output_dir,
        raw_config=config,
        resume_from=resume_path,
    )

    test_metrics = results.get("test_metrics")
    if test_metrics:
        print("[test metrics]")
        print(format_metrics_table(test_metrics))


if __name__ == "__main__":
    main()
