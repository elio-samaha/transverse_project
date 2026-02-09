from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _to_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _single_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            cm[true_idx, pred_idx] += 1

    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    weighted_f1_acc = 0.0
    total_support = int(cm.sum())

    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        support = int(cm[idx, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_acc += f1 * support

    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = _safe_div(weighted_f1_acc, total_support)
    accuracy = _safe_div(int(np.trace(cm)), total_support)

    tp_sum = int(np.trace(cm))
    fp_sum = int(cm.sum(axis=0).sum() - tp_sum)
    fn_sum = int(cm.sum(axis=1).sum() - tp_sum)
    micro_precision = _safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = _safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "task_type": "single_label",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "total_support": total_support,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"Multilabel metrics expect 2D tensors. Got y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    num_classes = len(class_names)
    if y_true.shape[1] != num_classes:
        raise ValueError(
            f"class_names length ({num_classes}) does not match target width ({y_true.shape[1]})"
        )

    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    total_tp = total_fp = total_fn = 0

    for idx, name in enumerate(class_names):
        true_col = y_true[:, idx].astype(np.int64)
        pred_col = y_pred[:, idx].astype(np.int64)
        tp = int(np.logical_and(true_col == 1, pred_col == 1).sum())
        fp = int(np.logical_and(true_col == 0, pred_col == 1).sum())
        fn = int(np.logical_and(true_col == 1, pred_col == 0).sum())
        tn = int(np.logical_and(true_col == 0, pred_col == 0).sum())
        support = int((true_col == 1).sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class.append(
            {
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_precision = float(np.mean(precision_values)) if precision_values else 0.0
    macro_recall = float(np.mean(recall_values)) if recall_values else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    exact_match_accuracy = float(np.all(y_true == y_pred, axis=1).mean()) if y_true.shape[0] else 0.0
    hamming_accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    return {
        "task_type": "multilabel",
        "accuracy": exact_match_accuracy,
        "hamming_accuracy": hamming_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_support": int(y_true.sum()),
        "per_class": per_class,
    }


def compute_classification_metrics(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
) -> dict[str, Any]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    if task_type in {"binary", "multiclass"}:
        return _single_label_metrics(
            y_true=y_true_np.astype(np.int64).reshape(-1),
            y_pred=y_pred_np.astype(np.int64).reshape(-1),
            class_names=class_names,
        )
    if task_type == "multilabel":
        return _multilabel_metrics(
            y_true=y_true_np.astype(np.int64),
            y_pred=y_pred_np.astype(np.int64),
            class_names=class_names,
        )
    raise ValueError(f"Unsupported task_type: {task_type}")


def format_metrics_table(metrics: dict[str, Any]) -> str:
    header = "Class                 Precision   Recall      F1         Support"
    line = "-" * len(header)
    rows = [header, line]
    for row in metrics.get("per_class", []):
        rows.append(
            f"{row['class_name'][:20]:20} "
            f"{row['precision']:10.4f} "
            f"{row['recall']:10.4f} "
            f"{row['f1']:10.4f} "
            f"{int(row['support']):10d}"
        )
    rows.extend(
        [
            line,
            f"accuracy: {metrics.get('accuracy', 0.0):.4f}",
            f"macro_f1: {metrics.get('macro_f1', 0.0):.4f}",
            f"macro_precision: {metrics.get('macro_precision', 0.0):.4f}",
            f"macro_recall: {metrics.get('macro_recall', 0.0):.4f}",
            f"micro_f1: {metrics.get('micro_f1', 0.0):.4f}",
        ]
    )
    return "\n".join(rows)
