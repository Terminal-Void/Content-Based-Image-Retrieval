"""以图搜图实验库。"""

from .core import (
    Feature,
    FeatureExtractor,
    FeatureGallery,
    GalleryEvaluation,
    QueryEvaluation,
    SearchResult,
    SimilarityMetric,
    evaluate_retrieval,
    list_image_paths,
    select_query_paths,
)
from .extractors import (
    DeepFeatureExtractor,
    HistogramFeatureExtractor,
    load_rgb_image,
)
from .visualization import (
    parent_category,
    plot_dataset_samples,
    plot_evaluation_comparison,
    plot_feature_illustration,
    plot_method_comparison,
    plot_search_results,
)

__all__ = [
    "DeepFeatureExtractor",
    "Feature",
    "FeatureExtractor",
    "FeatureGallery",
    "GalleryEvaluation",
    "HistogramFeatureExtractor",
    "QueryEvaluation",
    "SearchResult",
    "SimilarityMetric",
    "evaluate_retrieval",
    "list_image_paths",
    "load_rgb_image",
    "parent_category",
    "plot_dataset_samples",
    "plot_evaluation_comparison",
    "plot_feature_illustration",
    "plot_method_comparison",
    "plot_search_results",
    "select_query_paths",
]
