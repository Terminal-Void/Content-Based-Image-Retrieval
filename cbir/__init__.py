"""基于内容的图像检索（CBIR）课程实验库"""

from .dataset import list_image_paths, select_query_paths
from .extractors import (
    DeepFeatureExtractor,
    Feature,
    FeatureExtractor,
    HistogramFeatureExtractor,
    load_rgb_image,
)
from .gallery import FeatureGallery, SearchResult
from .visualization import plot_method_comparison, plot_search_results

__all__ = [
    "DeepFeatureExtractor",
    "Feature",
    "FeatureExtractor",
    "FeatureGallery",
    "HistogramFeatureExtractor",
    "SearchResult",
    "list_image_paths",
    "load_rgb_image",
    "plot_method_comparison",
    "plot_search_results",
    "select_query_paths",
]
