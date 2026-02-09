from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass
class DatasetConfig:
    task_type: str
    num_frames: int
    image_size: int
    clip_duration_sec: float | None
    random_clip_for_train: bool


class VideoClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        split: str,
        config: DatasetConfig,
    ) -> None:
        self.records = records
        self.class_to_idx = class_to_idx
        self.split = split
        self.config = config
        self._duration_cache: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _probe_duration(self, video_path: str) -> float:
        cached = self._duration_cache.get(video_path)
        if cached is not None:
            return cached

        duration = 0.0
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration = float(container.duration / av.time_base)
        except Exception:
            duration = 0.0

        self._duration_cache[video_path] = duration
        return duration

    def _compute_clip_window(self, duration: float) -> tuple[float, float | None]:
        clip_sec = self.config.clip_duration_sec
        if clip_sec is None or duration <= 0:
            return 0.0, None
        if duration <= clip_sec:
            return 0.0, None

        max_start = duration - clip_sec
        if self.split == "train" and self.config.random_clip_for_train:
            start = random.uniform(0.0, max_start)
        else:
            start = max_start / 2.0
        return start, start + clip_sec

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # Input: (T, C, H, W), output: (num_frames, C, H, W)
        t = frames.shape[0]
        target = self.config.num_frames
        if t == target:
            return frames
        if t > target:
            idx = torch.linspace(0, t - 1, steps=target).long()
            return frames.index_select(0, idx)

        repeats = (target + t - 1) // t
        expanded = frames.repeat((repeats, 1, 1, 1))
        return expanded[:target]

    def _load_video_tensor(self, video_path: str, duration_hint: float | None) -> torch.Tensor:
        duration = duration_hint if duration_hint is not None else self._probe_duration(video_path)
        start_sec, end_sec = self._compute_clip_window(duration)

        try:
            frames, _, _ = read_video(
                video_path,
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit="sec",
                output_format="TCHW",
            )
        except Exception:
            frames = torch.empty((0, 3, self.config.image_size, self.config.image_size), dtype=torch.uint8)

        if frames.numel() == 0:
            frames, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")

        if frames.numel() == 0:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        frames = self._sample_frames(frames)
        frames = frames.float() / 255.0
        frames = F.interpolate(
            frames,
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        # (T, C, H, W) -> (C, T, H, W) for torchvision video models.
        return frames.permute(1, 0, 2, 3).contiguous()

    def _build_target(self, record: dict[str, Any]) -> torch.Tensor:
        if self.config.task_type in {"binary", "multiclass"}:
            label = str(record["label"])
            return torch.tensor(self.class_to_idx[label], dtype=torch.long)

        # Multilabel mode.
        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in record.get("labels", []):
            if label not in self.class_to_idx:
                raise KeyError(
                    f"Unknown label '{label}' in record '{record.get('id', 'unknown')}'. "
                    "Update class_names in config or fix manifest labels."
                )
            target[self.class_to_idx[label]] = 1.0
        return target

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        video_path = str(Path(record["video_path"]).resolve())
        video = self._load_video_tensor(video_path, record.get("duration_sec"))
        target = self._build_target(record)
        return {
            "pixel_values": video,
            "labels": target,
            "video_path": video_path,
            "id": record.get("id", Path(video_path).stem),
        }
