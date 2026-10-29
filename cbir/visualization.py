"""以图搜图结果可视化。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .core import SearchResult

CategoryResolver = Callable[[Path], str]

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def parent_category(image_path: str | Path) -> str:
    """将图片的父目录作为类别名称。"""
    return Path(image_path).parent.name or "未标注"


def _get_category(
    image_path: str | Path,
    category: str | None,
    category_resolver: CategoryResolver | None,
) -> str:
    if category:
        return category
    if category_resolver is not None:
        return category_resolver(Path(image_path))
    return "未标注"


def _read_image(image_path: str | Path):
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def plot_search_results(
    query_path: str | Path,
    results: Sequence[SearchResult],
    feature_method: str,
    query_category: str | None = None,
    category_resolver: CategoryResolver | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """展示查询图片和前五名检索结果。

    图中包含特征方法、查询图片类别、结果排名、结果类别和相似度。
    ``category_resolver`` 适用于类别由目录结构提供的情况；如果没有类别
    元数据，图中会明确显示“未标注”。
    """
    query_path = Path(query_path)
    top_results = list(results)[:5]
    image_count = len(top_results) + 1
    figure, axes = plt.subplots(
        1,
        image_count,
        figsize=(4.0 * image_count, 4.8),
        squeeze=False,
    )
    axes = axes[0]

    query_axis = axes[0]
    query_axis.imshow(_read_image(query_path))
    query_axis.set_title(
        f"查询图片\n类别：{_get_category(query_path, query_category, category_resolver)}\n"
        f"{query_path.name}",
        fontsize=10,
    )
    query_axis.axis("off")

    for axis, result in zip(axes[1:], top_results):
        result_path = Path(result.path)
        axis.imshow(_read_image(result_path))
        category = _get_category(
            result_path,
            result.category,
            category_resolver,
        )
        axis.set_title(
            f"Top-{result.rank}\n类别：{category}\n"
            f"相似度：{result.score:.4f}\n{result_path.name}",
            fontsize=10,
        )
        axis.axis("off")

    figure.suptitle(
        f"以图搜图结果｜特征方法：{feature_method}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure
