from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_classification_metrics
from .utils import ensure_dir, save_json, to_device

try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast

    def _make_grad_scaler(enabled: bool) -> AmpGradScaler:
        return AmpGradScaler("cuda", enabled=enabled)

    def _autocast(enabled: bool):
        return amp_autocast(device_type="cuda", enabled=enabled)

except Exception:  # pragma: no cover - fallback for older torch versions.
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast

    def _make_grad_scaler(enabled: bool) -> AmpGradScaler:
        return AmpGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        return amp_autocast(enabled=enabled)


@dataclass
class TrainConfig:
    task_type: str
    epochs: int
    amp: bool
    grad_clip_norm: float | None
    log_every: int
    multilabel_threshold: float


def build_loss_fn(task_type: str) -> nn.Module:
    if task_type in {"binary", "multiclass"}:
        return nn.CrossEntropyLoss()
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported task type: {task_type}")


def _forward_loss(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(batch["pixel_values"])
    labels = batch["labels"]
    if task_type == "multilabel":
        loss = criterion(logits, labels.float())
    else:
        loss = criterion(logits, labels.long())
    return logits, loss


def _to_predictions(
    logits: torch.Tensor,
    task_type: str,
    threshold: float,
) -> torch.Tensor:
    if task_type == "multilabel":
        probs = torch.sigmoid(logits)
        return (probs >= threshold).long()
    return torch.argmax(logits, dim=1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    scaler: AmpGradScaler | None,
    epoch_index: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    batches = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(enabled=config.amp):
            _, loss = _forward_loss(model, batch, criterion, config.task_type)

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.item())
        batches += 1

        if config.log_every > 0 and step % config.log_every == 0:
            avg_loss = running_loss / max(batches, 1)
            print(
                f"[train] epoch={epoch_index} step={step}/{len(loader)} "
                f"loss={avg_loss:.4f}"
            )

    elapsed = time.time() - start
    return {
        "loss": running_loss / max(batches, 1),
        "seconds": elapsed,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
    class_names: list[str],
    multilabel_threshold: float,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    batches = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    for batch in loader:
        batch = to_device(batch, device)
        logits, loss = _forward_loss(model, batch, criterion, task_type)
        preds = _to_predictions(logits, task_type=task_type, threshold=multilabel_threshold)

        all_targets.append(batch["labels"].detach().cpu())
        all_predictions.append(preds.detach().cpu())
        running_loss += float(loss.item())
        batches += 1

    if not all_targets:
        raise ValueError("Evaluation loader is empty. Provide at least one sample.")

    y_true = torch.cat(all_targets, dim=0)
    y_pred = torch.cat(all_predictions, dim=0)
    metrics = compute_classification_metrics(task_type, y_true, y_pred, class_names)
    metrics["loss"] = running_loss / max(batches, 1)
    return metrics


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: AmpGradScaler | None,
    best_metric: float,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    is_best: bool,
) -> None:
    ensure_dir(output_dir)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "config": config,
    }
    epoch_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    last_path = output_dir / "checkpoint_last.pt"
    torch.save(checkpoint, epoch_path)
    shutil.copy2(epoch_path, last_path)

    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        shutil.copy2(epoch_path, best_path)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def try_load_checkpoint(
    checkpoint_path: str | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: AmpGradScaler | None = None,
    map_location: str = "cpu",
    optimizer_device: torch.device | None = None,
) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - for older torch.
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if optimizer_device is not None:
            _move_optimizer_state_to_device(optimizer, optimizer_device)
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    train_config: TrainConfig,
    class_names: list[str],
    output_dir: str | Path,
    raw_config: dict[str, Any],
    resume_from: str | None = None,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    criterion = build_loss_fn(train_config.task_type)
    scaler = _make_grad_scaler(enabled=train_config.amp)

    start_epoch = 1
    best_metric = float("-inf")

    checkpoint = try_load_checkpoint(
        resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device.type,
        optimizer_device=device,
    )
    if checkpoint is not None:
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", float("-inf")))
        print(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

    model.to(device)
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, train_config.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=train_config,
            scaler=scaler,
            epoch_index=epoch,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task_type=train_config.task_type,
            class_names=class_names,
            multilabel_threshold=train_config.multilabel_threshold,
        )
        if scheduler is not None:
            scheduler.step()

        tracked_metric = float(val_metrics.get("macro_f1", 0.0))
        is_best = tracked_metric > best_metric
        if is_best:
            best_metric = tracked_metric

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        save_checkpoint(
            output_dir=output_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_metric,
            class_to_idx=class_to_idx,
            config=raw_config,
            is_best=is_best,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_seconds": train_stats["seconds"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "best_macro_f1": best_metric,
            "is_best": is_best,
        }
        history.append(epoch_summary)
        print(
            f"[epoch {epoch}] train_loss={epoch_summary['train_loss']:.4f} "
            f"val_loss={epoch_summary['val_loss']:.4f} "
            f"val_acc={epoch_summary['val_accuracy']:.4f} "
            f"val_macro_f1={epoch_summary['val_macro_f1']:.4f}"
        )
        save_json(val_metrics, output_path / f"metrics_val_epoch_{epoch:03d}.json")
        save_json({"history": history}, output_path / "history.json")

    final_payload: dict[str, Any] = {"history": history}
    if test_loader is not None:
        best_checkpoint = output_path / "checkpoint_best.pt"
        if best_checkpoint.exists():
            try:
                payload = torch.load(best_checkpoint, map_location=device.type, weights_only=False)
            except TypeError:  # pragma: no cover - for older torch.
                payload = torch.load(best_checkpoint, map_location=device.type)
            model.load_state_dict(payload["model_state_dict"])
            model.to(device)
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            task_type=train_config.task_type,
            class_names=class_names,
            multilabel_threshold=train_config.multilabel_threshold,
        )
        save_json(test_metrics, output_path / "metrics_test.json")
        final_payload["test_metrics"] = test_metrics

    return final_payload
