"""特征对象、特征提取器接口和特征库"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一条检索结果"""
    rank: int
    score: float
    path: str
    category: str | None = None


class FeatureExtractor(ABC):
    """特征提取器抽象基类所有特征提取器继承此类并实现 ``_extract_vector`` 方法"""

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
        """提取并返回一维归一化特征向量"""
        raise NotImplementedError

    def extract(self, image_path: str | Path) -> Feature:
        """提取图像特征，并附加提取器类型和源文件路径"""
        return Feature(
            vector=self._extract_vector(image_path),
            extractor_cls=type(self),
            source_path=str(image_path),
        )

    def extract_many(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """批量提取特征；具体提取器可以覆盖此方法进行加速"""
        return np.asarray(
            [self.extract(path).vector for path in image_paths],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class Feature:
    """带有来源信息和提取器类型的特征向量"""

    vector: np.ndarray
    extractor_cls: type[FeatureExtractor]
    source_path: str | None = None

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

    def __matmul__(self, other: Feature) -> float:
        """使用点积计算两个已归一化特征向量的余弦相似度"""
        if not isinstance(other, Feature):
            raise TypeError(f"无法与非 Feature 对象计算相似度: {type(other)}")
        if self.extractor_cls is not other.extractor_cls:
            raise TypeError(
                f"无法在 '{self.feature_name}' 和 '{other.feature_name}' 之间比对！"
            )
        if self.vector.shape != other.vector.shape:
            raise ValueError("两个特征向量的维度必须一致")
        return float(np.dot(self.vector, other.vector))


@dataclass(slots=True)
class FeatureGallery:
    """特征库，支持缓存和批量相似度检索"""

    features: np.ndarray
    paths: list[str]
    extractor_cls: type[FeatureExtractor]
    categories: list[str | None] | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        paths = [str(path) for path in self.paths]
        if features.ndim != 2:
            raise ValueError("features 必须是二维特征矩阵")
        if len(features) != len(paths):
            raise ValueError("features 与 paths 的数量必须一致")
        if self.categories is None:
            categories = [None] * len(paths)
        else:
            categories = [category or None for category in self.categories]
            if len(categories) != len(paths):
                raise ValueError("categories 与 paths 的数量必须一致")
        self.features = features
        self.paths = paths
        self.categories = categories

    def search(self, query: Feature, top_k: int = 5) -> list[SearchResult]:
        """返回与查询特征最相似的前 ``top_k`` 张图片"""
        if top_k < 1:
            raise ValueError("top_k 必须是正整数")
        if query.extractor_cls is not self.extractor_cls:
            raise TypeError(
                f"试图用查询图特征【{query.feature_name}】检索"
                f"【{self.extractor_cls.__name__}】特征库！"
            )
        if self.features.shape[1] != query.vector.shape[0]:
            raise ValueError("查询特征与特征库的向量维度不一致")

        scores = np.dot(self.features, query.vector)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                rank=rank,
                score=float(scores[index]),
                path=self.paths[index],
                category=self.categories[index],
            )
            for rank, index in enumerate(top_indices, start=1)
        ]

    def save(self, file_path: str | Path) -> None:
        """将特征矩阵、图片路径和提取器名称保存为压缩文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            features=self.features,
            paths=np.asarray(self.paths, dtype=str),
            categories=np.asarray(
                [category or "" for category in self.categories],
                dtype=str,
            ),
            extractor_name=self.extractor_cls.__name__,
        )

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        expected_extractor: type[FeatureExtractor] | None = None,
    ) -> FeatureGallery:
        """从缓存文件加载特征库，并校验特征提取器类型"""
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
            paths = [str(path) for path in data["paths"]]
            if "categories" in data.files:
                categories = [str(category) or None for category in data["categories"]]
            else:
                categories = None

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
        categories: Sequence[str | None] | None = None,
    ) -> FeatureGallery:
        """优先加载匹配缓存，否则提取特征并可选地保存缓存"""
        normalized_paths = [str(path) for path in image_paths]
        if not normalized_paths:
            raise ValueError("image_paths 不能为空")
        normalized_categories = (
            [category or None for category in categories]
            if categories is not None
            else None
        )
        if normalized_categories is not None and len(normalized_categories) != len(
            normalized_paths
        ):
            raise ValueError("categories 与 image_paths 的数量必须一致")

        if cache_path and Path(cache_path).exists():
            cached_gallery = cls.load(
                cache_path,
                expected_extractor=type(extractor),
            )
            categories_match = (
                normalized_categories is None
                or cached_gallery.categories == normalized_categories
            )
            if cached_gallery.paths == normalized_paths and categories_match:
                return cached_gallery

        features = extractor.extract_many(normalized_paths)
        gallery = cls(
            features=features,
            paths=normalized_paths,
            extractor_cls=type(extractor),
            categories=normalized_categories,
        )
        if cache_path:
            gallery.save(cache_path)
        return gallery
