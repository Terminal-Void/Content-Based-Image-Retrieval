"""特征对象、特征提取器接口、特征库检索与评测指标计算。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SimilarityMetric = Literal["cosine", "euclidean", "l1", "intersection"]


def list_image_paths(root: str | Path) -> list[Path]:
    """扫描按类别子目录组织的图片路径，忽略隐藏文件与隐藏目录。"""
    root_path = Path(root)
    if not root_path.exists():
        return []
    paths: list[Path] = []
    for category_dir in sorted(root_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        for file_path in sorted(category_dir.iterdir()):
            if (
                file_path.is_file()
                and not file_path.name.startswith(".")
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                paths.append(file_path)
    return paths


def select_query_paths(
    query_root: str | Path,
    per_category: int = 2,
) -> list[Path]:
    """从每个类别子目录中按顺序选取前 ``per_category`` 张图片作为查询集。"""
    if per_category < 1:
        raise ValueError("per_category 必须是正整数")
    root_path = Path(query_root)
    if not root_path.exists():
        return []
    selected: list[Path] = []
    for category_dir in sorted(root_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        category_files = [
            file_path
            for file_path in sorted(category_dir.iterdir())
            if (
                file_path.is_file()
                and not file_path.name.startswith(".")
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
        selected.extend(category_files[:per_category])
    return selected


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一条检索结果。"""

    rank: int
    score: float
    path: str
    category: str | None = None
    metric: str = "cosine"


class FeatureExtractor(ABC):
    """特征提取器抽象基类，所有提取器继承此类并实现 ``_extract_vector``。"""

    _registry: ClassVar[dict[str, type[FeatureExtractor]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def get_extractor_class(cls, name: str) -> type[FeatureExtractor] | None:
        """根据类名查找已注册的特征提取器。"""
        return cls._registry.get(name)

    @abstractmethod
    def _extract_vector(self, image_path: str | Path) -> np.ndarray:
        """提取并返回一维归一化特征向量。"""
        raise NotImplementedError

    def extract(self, image_path: str | Path) -> Feature:
        """提取图像特征，并附加提取器类型和源文件路径。"""
        return Feature(
            vector=self._extract_vector(image_path),
            extractor_cls=type(self),
            source_path=str(image_path),
        )

    def extract_many(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """批量提取特征；具体提取器可以覆盖此方法进行加速。"""
        return np.asarray(
            [self.extract(path).vector for path in image_paths],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class Feature:
    """带有来源信息和提取器类型的特征向量。"""

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
        """返回特征提取器名称。"""
        return self.extractor_cls.__name__

    def __matmul__(self, other: Feature) -> float:
        """使用点积计算两个已归一化特征向量的余弦相似度。"""
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
    """特征库，支持缓存和批量相似度检索。"""

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

    def _compute_scores(
        self,
        query_vector: np.ndarray,
        metric: SimilarityMetric = "cosine",
    ) -> tuple[np.ndarray, bool]:
        """计算特征库与查询特征的度量值，返回 (scores, higher_is_better)。"""
        if metric == "cosine":
            return np.dot(self.features, query_vector), True
        if metric == "euclidean":
            distances = np.linalg.norm(self.features - query_vector, axis=1)
            return distances, False
        if metric == "l1":
            distances = np.abs(self.features - query_vector).sum(axis=1)
            return distances, False
        if metric == "intersection":
            g_sum = np.maximum(self.features.sum(axis=1, keepdims=True), 1e-12)
            q_sum = max(float(query_vector.sum()), 1e-12)
            g_prob = self.features / g_sum
            q_prob = query_vector / q_sum
            return np.minimum(g_prob, q_prob).sum(axis=1), True
        raise ValueError(f"不支持的度量方式: {metric}")

    def search(
        self,
        query: Feature,
        top_k: int = 5,
        metric: SimilarityMetric = "cosine",
    ) -> list[SearchResult]:
        """返回与查询特征最相似的前 ``top_k`` 张图片。"""
        if top_k < 1:
            raise ValueError("top_k 必须是正整数")
        if query.extractor_cls is not self.extractor_cls:
            raise TypeError(
                f"试图用查询图特征【{query.feature_name}】检索"
                f"【{self.extractor_cls.__name__}】特征库！"
            )
        if self.features.shape[1] != query.vector.shape[0]:
            raise ValueError("查询特征与特征库的向量维度不一致")

        scores, higher_is_better = self._compute_scores(query.vector, metric=metric)
        order = np.argsort(scores)
        if higher_is_better:
            order = order[::-1]
        top_indices = order[:top_k]

        return [
            SearchResult(
                rank=rank,
                score=float(scores[index]),
                path=self.paths[index],
                category=self.categories[index],
                metric=metric,
            )
            for rank, index in enumerate(top_indices, start=1)
        ]

    def save(self, file_path: str | Path) -> None:
        """将特征矩阵、图片路径和提取器名称保存为压缩文件。"""
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
        """从缓存文件加载特征库，并校验特征提取器类型。"""
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
        """优先加载匹配缓存，否则提取特征并可选地保存缓存。"""
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
            expected_dim = getattr(extractor, "feature_dim", None)
            dim_match = (
                expected_dim is None
                or cached_gallery.features.shape[1] == expected_dim
            )
            if (
                cached_gallery.paths == normalized_paths
                and categories_match
                and dim_match
            ):
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


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """单张查询图的检索评测记录。"""

    query_path: Path
    query_category: str
    results: list[SearchResult]
    top1_hit: bool
    hits_at_k: int
    precision_at_k: float
    ap_at_k: float


@dataclass(frozen=True, slots=True)
class GalleryEvaluation:
    """整个查询集在指定特征库上的综合评测结果。"""

    method_name: str
    top_k: int
    queries: list[QueryEvaluation]
    top1_accuracy: float
    hit_rate_at_k: float
    mean_precision_at_k: float
    map_at_k: float
    category_precision: dict[str, float]


def _average_precision_at_k(hits: Sequence[bool], top_k: int) -> float:
    """计算截断在 K 处的 Average Precision (AP@K)。"""
    if not any(hits):
        return 0.0
    running_hits = 0
    precision_sum = 0.0
    for idx, is_hit in enumerate(hits[:top_k], start=1):
        if is_hit:
            running_hits += 1
            precision_sum += running_hits / idx
    return precision_sum / running_hits


def evaluate_retrieval(
    gallery: FeatureGallery,
    query_paths: Sequence[str | Path],
    extractor: FeatureExtractor,
    method_name: str,
    top_k: int = 5,
    metric: SimilarityMetric = "cosine",
) -> GalleryEvaluation:
    """对一组查询图片执行检索并统计 Top-1 准确率、Precision@K、Hit@K 与 mAP@K。"""
    paths = [Path(p) for p in query_paths]
    query_vectors = extractor.extract_many(paths)

    query_evals: list[QueryEvaluation] = []
    cat_precisions: dict[str, list[float]] = {}

    for path, vec in zip(paths, query_vectors):
        query_cat = path.parent.name
        feature = Feature(
            vector=vec,
            extractor_cls=type(extractor),
            source_path=str(path),
        )
        results = gallery.search(feature, top_k=top_k, metric=metric)
        hits = [r.category == query_cat for r in results]
        hits_count = sum(hits)
        p_at_k = hits_count / top_k
        ap_at_k = _average_precision_at_k(hits, top_k=top_k)

        query_evals.append(
            QueryEvaluation(
                query_path=path,
                query_category=query_cat,
                results=results,
                top1_hit=bool(hits[0]) if hits else False,
                hits_at_k=hits_count,
                precision_at_k=p_at_k,
                ap_at_k=ap_at_k,
            )
        )
        cat_precisions.setdefault(query_cat, []).append(p_at_k)

    n_queries = max(len(query_evals), 1)
    top1_acc = sum(1 for q in query_evals if q.top1_hit) / n_queries
    hit_rate = sum(1 for q in query_evals if q.hits_at_k > 0) / n_queries
    mean_p = float(np.mean([q.precision_at_k for q in query_evals]))
    mean_ap = float(np.mean([q.ap_at_k for q in query_evals]))
    category_precision = {
        cat: float(np.mean(vals)) for cat, vals in sorted(cat_precisions.items())
    }

    return GalleryEvaluation(
        method_name=method_name,
        top_k=top_k,
        queries=query_evals,
        top1_accuracy=top1_acc,
        hit_rate_at_k=hit_rate,
        mean_precision_at_k=mean_p,
        map_at_k=mean_ap,
        category_precision=category_precision,
    )
