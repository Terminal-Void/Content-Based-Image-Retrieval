"""检索结果数据结构与特征检索库（支持类型校验、相似度排序与本地缓存）"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, assert_never

import numpy as np

from .extractors import Feature, FeatureExtractor

MetricType = Literal["cosine", "euclidean"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """单条检索结果"""

    rank: int
    score: float
    path: str
    category: str
    metric: MetricType = "cosine"


@dataclass
class FeatureGallery:
    """图像特征检索库，支持提取器类型校验、Top-K 检索与 .npz 缓存读写"""

    features: np.ndarray
    paths: list[str]
    extractor_cls: type[FeatureExtractor]
    categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        paths = [str(p) for p in self.paths]
        if features.ndim != 2:
            raise ValueError("features 必须是二维特征矩阵 (N, D)")
        if len(features) != len(paths):
            raise ValueError("features 与 paths 的数量必须一致")
        if not self.categories:
            categories = [Path(p).parent.name for p in paths]
        else:
            if len(self.categories) != len(paths):
                raise ValueError("categories 与 paths 的数量必须一致")
            categories = [str(c) for c in self.categories]

        self.features = features
        self.paths = paths
        self.categories = categories

    def search(
        self,
        query: Feature,
        top_k: int = 5,
        metric: MetricType = "cosine",
    ) -> list[SearchResult]:
        """校验提取器类型并计算相似度或欧氏距离，返回排名前 top_k 的图片"""
        if top_k < 1:
            raise ValueError("top_k 必须是正整数")
        if query.extractor_cls is not self.extractor_cls:
            raise TypeError(
                f"试图用查询图特征 {query.feature_name} 检索"
                f" {self.extractor_cls.__name__} 特征库！"
            )
        if self.features.shape[1] != query.vector.shape[0]:
            raise ValueError("查询特征与特征库的向量维度不一致")

        if metric == "cosine":
            scores = self.features @ query.vector
            top_indices = np.argsort(scores)[::-1][:top_k]
        elif metric == "euclidean":
            scores = np.linalg.norm(self.features - query.vector, axis=1)
            top_indices = np.argsort(scores)[:top_k]
        else:
            assert_never(metric)

        return [
            SearchResult(
                rank=rank,
                score=float(scores[idx]),
                path=self.paths[idx],
                category=self.categories[idx],
                metric=metric,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

    def save(self, file_path: str | Path) -> None:
        """将特征矩阵、图片路径和提取器类名保存为压缩 .npz 文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            features=self.features,
            paths=np.asarray(self.paths, dtype=str),
            categories=np.asarray(self.categories, dtype=str),
            extractor_name=self.extractor_cls.__name__,
        )

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        expected_extractor: type[FeatureExtractor] | None = None,
    ) -> FeatureGallery:
        """从 .npz 缓存文件加载特征库，并通过注册表恢复和校验特征提取器类"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"未找到特征库缓存文件: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            extractor_name = str(data["extractor_name"])
            recovered_cls = FeatureExtractor.get_extractor_class(extractor_name)
            if recovered_cls is None:
                raise RuntimeError(
                    f"无法找到名为 '{extractor_name}' 的提取器类，"
                    "请先导入对应的特征提取器"
                )
            if expected_extractor and recovered_cls is not expected_extractor:
                raise TypeError(
                    f"缓存类型为 '{extractor_name}'，"
                    f"但预期类型为 '{expected_extractor.__name__}'"
                )

            features = data["features"].copy()
            paths = [str(p) for p in data["paths"]]
            categories = (
                [str(c) for c in data["categories"]]
                if "categories" in data.files
                else [Path(p).parent.name for p in paths]
            )

        return cls(
            features=features,
            paths=paths,
            extractor_cls=recovered_cls,
            categories=categories,
        )

    @classmethod
    def build_or_load(
        cls,
        image_paths: Sequence[str | Path],
        extractor: FeatureExtractor,
        cache_path: str | Path | None = None,
        categories: Sequence[str] | None = None,
    ) -> FeatureGallery:
        """若存在匹配缓存则直接加载，否则提取特征并可选地写入缓存"""
        normalized_paths = [str(p) for p in image_paths]
        if not normalized_paths:
            raise ValueError("image_paths 不能为空")
        resolved_categories = (
            [str(c) for c in categories]
            if categories is not None
            else [Path(p).parent.name for p in normalized_paths]
        )

        if cache_path and Path(cache_path).exists():
            cached = cls.load(cache_path, expected_extractor=type(extractor))
            if cached.paths == normalized_paths and cached.categories == resolved_categories:
                return cached

        features = extractor.extract_many(normalized_paths)
        gallery = cls(
            features=features,
            paths=normalized_paths,
            extractor_cls=type(extractor),
            categories=resolved_categories,
        )
        if cache_path:
            gallery.save(cache_path)
        return gallery
