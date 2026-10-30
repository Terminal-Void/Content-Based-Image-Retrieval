"""图像读取预处理、颜色直方图特征提取器与深度特征提取器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models

from .core import FeatureExtractor


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """将向量归一化为单位长度，便于使用点积计算余弦相似度。"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("特征向量范数为 0，无法归一化")
    return vector / norm


def load_rgb_image(
    image_path: str | Path,
    resize: tuple[int, int] | None = None,
) -> Image.Image:
    """读取图像并进行基础预处理：EXIF 方向校正、统一转换为 RGB 通道及可选缩放。"""
    with Image.open(image_path) as image:
        corrected = ImageOps.exif_transpose(image)
        rgb_image = corrected.convert("RGB")
        if resize is not None:
            rgb_image = rgb_image.resize(resize, Image.Resampling.BILINEAR)
        return rgb_image.copy()


class HistogramFeatureExtractor(FeatureExtractor):
    """提取 RGB 三通道归一化颜色直方图特征。"""

    def __init__(
        self,
        bins_per_channel: int = 16,
        resize: tuple[int, int] | None = None,
    ) -> None:
        if bins_per_channel < 1:
            raise ValueError("bins_per_channel 必须是正整数")
        self.bins_per_channel = bins_per_channel
        self.resize = resize

    @property
    def feature_dim(self) -> int:
        """拼接后的特征向量维度（3 × bins_per_channel）。"""
        return 3 * self.bins_per_channel

    def extract_channel_histograms(self, image_path: str | Path) -> np.ndarray:
        """提取 R、G、B 三个通道各自归一化后的概率分布，返回形状为 (3, bins)。"""
        rgb_image = np.asarray(
            load_rgb_image(image_path, resize=self.resize),
            dtype=np.uint8,
        )
        channel_histograms = []
        for channel in range(3):
            histogram, _ = np.histogram(
                rgb_image[:, :, channel],
                bins=self.bins_per_channel,
                range=(0, 256),
            )
            histogram = histogram.astype(np.float32)
            total = histogram.sum()
            if total > 0:
                histogram /= total
            channel_histograms.append(histogram)
        return np.stack(channel_histograms, axis=0)

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        channel_histograms = self.extract_channel_histograms(image_path)
        vector = channel_histograms.reshape(-1)
        return _l2_normalize(vector).astype(np.float32)


class DeepFeatureExtractor(FeatureExtractor):
    """使用 ImageNet 预训练 ResNet-50 全局平均池化层输出作为深度特征。"""

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

    @property
    def feature_dim(self) -> int:
        """ResNet-50 倒数第二层（全局平均池化层）输出维度。"""
        return 2048

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        rgb_image = load_rgb_image(image_path)
        input_tensor = self.preprocess(rgb_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model(input_tensor)
        vector = features.cpu().numpy().reshape(-1).astype(np.float32)
        return _l2_normalize(vector).astype(np.float32)

    def extract_many(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        """分批提取深度特征，减少模型前向调用次数。"""
        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")

        paths = list(image_paths)
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = [
                self.preprocess(load_rgb_image(path)) for path in batch_paths
            ]
            input_batch = torch.stack(tensors).to(self.device)

            with torch.inference_mode():
                features = self.model(input_batch)
            vectors = features.cpu().numpy().reshape(len(batch_paths), -1)
            batches.append(
                np.asarray([_l2_normalize(vector) for vector in vectors])
            )

        if not batches:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.concatenate(batches).astype(np.float32)
