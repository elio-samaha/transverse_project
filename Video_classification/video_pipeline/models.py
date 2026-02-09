from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)


SUPPORTED_TORCHVISION_MODELS = {"r3d_18", "mc3_18", "r2plus1d_18"}
SUPPORTED_EXPERIMENTAL_VLM_MODELS = {"internvideo2_5", "videollama3", "internvl", "hf_backbone"}
SUPPORTED_MMACTION2_MODELS = {"mmaction2"}


DEFAULT_HF_MODEL_IDS = {
    "internvideo2_5": "OpenGVLab/InternVideo2_5_Chat_8B",
    "videollama3": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "internvl": "OpenGVLab/InternVL2_5-8B",
}


@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    dropout: float
    freeze_backbone: bool
    hf_model_id: str | None = None
    hf_trust_remote_code: bool = True
    mmaction_config_path: str | None = None
    mmaction_checkpoint_path: str | None = None
    mmaction_load_strict: bool = False


def _freeze_all_except_classifier(module: nn.Module, classifier_names: tuple[str, ...]) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False

    for name in classifier_names:
        layer = getattr(module, name, None)
        if layer is None:
            continue
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _build_torchvision_model(config: ModelConfig, num_outputs: int) -> nn.Module:
    name = config.name
    if name == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if config.pretrained else None
        model = r3d_18(weights=weights)
    elif name == "mc3_18":
        weights = MC3_18_Weights.DEFAULT if config.pretrained else None
        model = mc3_18(weights=weights)
    elif name == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.DEFAULT if config.pretrained else None
        model = r2plus1d_18(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision model: {name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=config.dropout),
        nn.Linear(in_features, num_outputs),
    )

    if config.freeze_backbone:
        _freeze_all_except_classifier(model, classifier_names=("fc",))

    return model


class HFVideoBackboneClassifier(nn.Module):
    """
    Experimental adapter for large video-language backbones.

    This assumes the backbone can consume a keyword argument such as `pixel_values`
    or `videos` and return tensor features that can be pooled into one vector.
    """

    def __init__(
        self,
        model_id: str,
        num_outputs: int,
        dropout: float,
        freeze_backbone: bool,
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as exc:
            raise ImportError(
                "transformers is required for experimental VLM backbones. "
                "Install with: pip install transformers"
            ) from exc

        self.backbone = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = self._infer_hidden_size(self.backbone)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden size for backbone {model_id}. "
                "Set a compatible model that returns pooled embeddings."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_outputs),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _infer_hidden_size(model: nn.Module) -> int | None:
        config = getattr(model, "config", None)
        if config is None:
            return None

        for key in ("hidden_size", "projection_dim", "d_model", "dim"):
            if hasattr(config, key):
                value = getattr(config, key)
                if isinstance(value, int):
                    return value

        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            for key in ("hidden_size", "projection_dim", "d_model", "dim"):
                value = getattr(vision_config, key, None)
                if isinstance(value, int):
                    return value

        return None

    @staticmethod
    def _pool_outputs(outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
            if hidden.ndim == 3:
                return hidden.mean(dim=1)
            return hidden.reshape(hidden.shape[0], -1)

        if isinstance(outputs, tuple) and outputs:
            first = outputs[0]
            if first.ndim == 3:
                return first.mean(dim=1)
            return first.reshape(first.shape[0], -1)

        raise ValueError("Backbone outputs do not expose a usable hidden representation.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Some backbones expect B,T,C,H,W, while torchvision-style tensors are B,C,T,H,W.
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(pixel_values.shape)}")

        btchw = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        last_error: Exception | None = None

        candidate_kwargs = (
            {"pixel_values": btchw},
            {"videos": btchw},
            {"video": btchw},
            {"pixel_values": pixel_values},
            {"videos": pixel_values},
        )
        for kwargs in candidate_kwargs:
            try:
                outputs = self.backbone(**kwargs, return_dict=True)
                features = self._pool_outputs(outputs)
                return self.classifier(features)
            except Exception as exc:  # pragma: no cover - depends on remote model API.
                last_error = exc
                continue

        raise RuntimeError(
            "Failed to run the selected VLM backbone with video tensors. "
            f"Last error: {last_error}"
        )


class MMAction2RecognizerClassifier(nn.Module):
    """
    Adapter for MMAction2 recognizer configs inside this training loop.

    Expected input tensor shape: B,C,T,H,W.
    """

    def __init__(
        self,
        num_outputs: int,
        config_path: str,
        checkpoint_path: str | None,
        dropout: float,
        freeze_backbone: bool,
        load_strict: bool,
    ) -> None:
        super().__init__()
        try:
            from mmengine.config import Config  # type: ignore
            from mmengine.runner import load_checkpoint  # type: ignore
            from mmaction.registry import MODELS  # type: ignore
        except Exception as exc:
            raise ImportError(
                "MMAction2 backend requires mmengine + mmaction2.\n"
                "Install with:\n"
                "  pip install -U openmim\n"
                "  mim install mmengine\n"
                "  mim install 'mmcv>=2.0.0'\n"
                "  mim install mmaction2"
            ) from exc

        cfg_path = Path(config_path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"MMAction2 config not found: {cfg_path}")
        cfg = Config.fromfile(str(cfg_path))
        if "model" not in cfg:
            raise ValueError(f"Invalid MMAction2 config (missing model): {cfg_path}")

        model_cfg = cfg.model.copy()
        cls_head = model_cfg.get("cls_head", None)
        if cls_head is None:
            raise ValueError(
                "MMAction2 model config must contain model.cls_head to set num_classes."
            )
        if "num_classes" in cls_head:
            cls_head["num_classes"] = num_outputs
        if "dropout_ratio" in cls_head:
            cls_head["dropout_ratio"] = dropout

        if not checkpoint_path:
            backbone_cfg = model_cfg.get("backbone", None)
            if isinstance(backbone_cfg, dict):
                backbone_cfg["pretrained"] = None
                backbone_cfg["init_cfg"] = None

        self.model = MODELS.build(model_cfg)

        if checkpoint_path:
            ckpt = str(Path(checkpoint_path).expanduser().resolve())
            load_checkpoint(self.model, ckpt, map_location="cpu", strict=load_strict)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for head_name in ("cls_head",):
                head = getattr(self.model, head_name, None)
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = True

    @staticmethod
    def _normalize_output(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            if output.ndim > 2:
                return output.reshape(output.shape[0], -1)
            return output

        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if torch.is_tensor(first):
                if first.ndim > 2:
                    return first.reshape(first.shape[0], -1)
                return first

        if isinstance(output, dict):
            for key in ("cls_scores", "logits", "pred_score", "scores"):
                value = output.get(key)
                if torch.is_tensor(value):
                    if value.ndim > 2:
                        return value.reshape(value.shape[0], -1)
                    return value

        raise ValueError(f"Unsupported MMAction2 output type: {type(output)}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(pixel_values.shape)}")

        # Most MMAction2 recognizers accept NCTHW in tensor mode.
        try:
            outputs = self.model(pixel_values, mode="tensor")
        except Exception:
            # Some configs expect an explicit clip dimension: B, num_clips, C, T, H, W.
            outputs = self.model(pixel_values.unsqueeze(1), mode="tensor")
        return self._normalize_output(outputs)


def build_model(config: ModelConfig, num_outputs: int) -> nn.Module:
    name = config.name.lower()
    if name in SUPPORTED_TORCHVISION_MODELS:
        return _build_torchvision_model(config, num_outputs=num_outputs)

    if name in SUPPORTED_MMACTION2_MODELS:
        if not config.mmaction_config_path:
            raise ValueError(
                "For model.name=mmaction2, set model.mmaction_config_path in your YAML config."
            )
        return MMAction2RecognizerClassifier(
            num_outputs=num_outputs,
            config_path=config.mmaction_config_path,
            checkpoint_path=config.mmaction_checkpoint_path,
            dropout=config.dropout,
            freeze_backbone=config.freeze_backbone,
            load_strict=config.mmaction_load_strict,
        )

    if name in SUPPORTED_EXPERIMENTAL_VLM_MODELS:
        model_id = config.hf_model_id or DEFAULT_HF_MODEL_IDS.get(name)
        if not model_id:
            raise ValueError(
                f"No model id configured for {name}. Provide model.hf_model_id in config."
            )
        return HFVideoBackboneClassifier(
            model_id=model_id,
            num_outputs=num_outputs,
            dropout=config.dropout,
            freeze_backbone=config.freeze_backbone,
            trust_remote_code=config.hf_trust_remote_code,
        )

    raise ValueError(
        f"Unknown model '{config.name}'. "
        f"Supported torchvision models: {sorted(SUPPORTED_TORCHVISION_MODELS)}. "
        f"Supported MMAction2 models: {sorted(SUPPORTED_MMACTION2_MODELS)}. "
        f"Experimental VLM models: {sorted(SUPPORTED_EXPERIMENTAL_VLM_MODELS)}."
    )
