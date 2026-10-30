"""图像读取预处理、特征对象、特征提取器抽象基类与具体实现"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """读取图像并完成基础预处理：EXIF 拍摄方向校正与 RGB 三通道转换"""
    with Image.open(image_path) as image:
        corrected = ImageOps.exif_transpose(image)
        return corrected.convert("RGB")


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    """对一维特征向量或二维特征矩阵按最后一维做 L2 归一化"""
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("特征向量范数为 0，无法进行 L2 归一化")
    return np.asarray(array / norms, dtype=np.float32)


class FeatureExtractor(ABC):
    """特征提取器抽象基类，子类定义时自动以 ``__name__`` 注册"""

    _registry: ClassVar[dict[str, type[FeatureExtractor]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def get_extractor_class(cls, name: str) -> type[FeatureExtractor] | None:
        """根据类名查找已注册的特征提取器"""
        return cls._registry.get(name)

    @abstractmethod
    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        """提取并返回一维 L2 归一化特征向量"""

    def extract(self, image_path: str | Path) -> Feature:
        """提取图像特征，并绑定提取器类型和源文件路径"""
        return Feature(
            vector=self._extract_vector(image_path),
            extractor_cls=type(self),
            source_path=str(image_path),
        )

    def extract_many(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """批量提取特征矩阵 (N, D)；子类可覆盖此方法进行批处理加速"""
        if not image_paths:
            raise ValueError("image_paths 不能为空")
        return np.stack([self._extract_vector(path) for path in image_paths]).astype(
            np.float32
        )


@dataclass(frozen=True, slots=True)
class Feature:
    """带有来源信息和提取器类型的特征向量"""

    vector: np.ndarray
    extractor_cls: type[FeatureExtractor]
    source_path: str = ""

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("特征向量必须是一维数组")
        if not np.isfinite(vector).all():
            raise ValueError("特征向量不能包含 NaN 或无穷值")
        object.__setattr__(self, "vector", vector)

    @property
    def feature_name(self) -> str:
        """返回特征提取器名称"""
        return self.extractor_cls.__name__

    def __matmul__(self, other: object) -> float:
        """使用点积计算两个同类已归一化特征向量的余弦相似度"""
        if not isinstance(other, Feature):
            raise TypeError(f"无法与非 Feature 对象计算相似度: {type(other)}")
        if self.extractor_cls is not other.extractor_cls:
            raise TypeError(
                f"无法在 '{self.feature_name}' 和 '{other.feature_name}' 之间比对！"
            )
        if self.vector.shape != other.vector.shape:
            raise ValueError("两个特征向量的维度必须一致")
        return float(np.dot(self.vector, other.vector))


class HistogramFeatureExtractor(FeatureExtractor):
    """提取 RGB 三通道归一化颜色直方图特征"""

    def __init__(self, bins_per_channel: int = 16) -> None:
        if bins_per_channel < 1:
            raise ValueError("bins_per_channel 必须是正整数")
        self.bins_per_channel = bins_per_channel

    def extract_channel_histograms(self, image_path: str | Path) -> np.ndarray:
        """提取 R、G、B 三通道各自的归一化概率直方图，返回形状为 (3, bins)"""
        rgb_image = np.asarray(load_rgb_image(image_path), dtype=np.uint8)
        channel_histograms: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(
                rgb_image[:, :, channel],
                bins=self.bins_per_channel,
                range=(0, 256),
            )
            hist_f32 = hist.astype(np.float32)
            hist_f32 /= hist_f32.sum()
            channel_histograms.append(hist_f32)
        return np.stack(channel_histograms, axis=0)

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        vector = self.extract_channel_histograms(image_path).reshape(-1)
        return _l2_normalize(vector)


class DeepFeatureExtractor(FeatureExtractor):
    """使用 ImageNet 预训练 ResNet-50 全局平均池化层输出提取深度特征"""

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                resolved_device: str | torch.device = "mps"
            elif torch.cuda.is_available():
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        base_model = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)
        self.preprocess = weights.transforms()

    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        return self.extract_many([image_path], batch_size=1)[0]

    def extract_many(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        """按 batch 前向推理提取深度特征矩阵"""
        if not image_paths:
            raise ValueError("image_paths 不能为空")
        if batch_size < 1:
            raise ValueError("batch_size 必须是正整数")

        paths = list(image_paths)
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors: list[torch.Tensor] = [
                self.preprocess(load_rgb_image(path)) for path in batch_paths
            ]
            input_batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                features = self.model(input_batch)
            vectors = features.cpu().numpy().reshape(len(batch_paths), -1)
            batches.append(_l2_normalize(vectors))

        return np.concatenate(batches, axis=0)
