"""颜色直方图和深度特征提取器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from .core import FeatureExtractor


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """将向量归一化为单位长度，便于使用点积计算余弦相似度。"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("特征向量范数为 0，无法归一化")
    return vector / norm


class HistogramFeatureExtractor(FeatureExtractor):
    """提取 RGB 三通道颜色直方图特征。"""

    def __init__(self, bins_per_channel: int = 16) -> None:
        if bins_per_channel < 1:
            raise ValueError("bins_per_channel 必须是正整数")
        self.bins_per_channel = bins_per_channel

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)

        channel_histograms = []
        for channel in range(3):
            histogram, _ = np.histogram(
                rgb_image[:, :, channel],
                bins=self.bins_per_channel,
                range=(0, 256),
            )
            histogram = histogram.astype(np.float32)
            histogram /= histogram.sum()
            channel_histograms.append(histogram)

        vector = np.concatenate(channel_histograms)
        return _l2_normalize(vector).astype(np.float32)


class DeepFeatureExtractor(FeatureExtractor):
    """使用预训练 ResNet-50 最后一层全连接层之前的输出作为特征。"""

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        base_model = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)
        self.preprocess = weights.transforms()

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            input_tensor = self.preprocess(image.convert("RGB"))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model(input_tensor)
        vector = features.cpu().numpy().reshape(-1).astype(np.float32)
        return _l2_normalize(vector).astype(np.float32)

    def extract_many(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        """分批提取深度特征，减少模型调用次数。"""
        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")

        paths = list(image_paths)
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            input_batch = torch.stack(tensors).to(self.device)

            with torch.inference_mode():
                features = self.model(input_batch)
            vectors = features.cpu().numpy().reshape(len(batch_paths), -1)
            batches.append(
                np.asarray([_l2_normalize(vector) for vector in vectors])
            )

        if not batches:
            return np.empty((0, 2048), dtype=np.float32)
        return np.concatenate(batches).astype(np.float32)
