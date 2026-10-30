"""以图搜图 Top-K 检索结果可视化"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, ImageOps

from .extractors import load_rgb_image
from .gallery import MetricType, SearchResult

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

METRIC_LABELS: dict[MetricType, str] = {
    "cosine": "余弦相似度",
    "euclidean": "欧氏距离",
}


def _read_canvas_image(
    image_path: str | Path,
    canvas_size: tuple[int, int] = (280, 280),
) -> Image.Image:
    """读取图片并等比例居中嵌入统一画布，保证多子图标题严格对齐"""
    image = load_rgb_image(image_path)
    return ImageOps.pad(
        image,
        canvas_size,
        method=Image.Resampling.BILINEAR,
        color=(246, 247, 249),
    )


def _style_border(axis: Axes, color: str, linewidth: float = 2.2) -> None:
    """设置子图边框颜色并隐藏坐标轴刻度"""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def plot_search_results(
    query_path: str | Path,
    results: Sequence[SearchResult],
    feature_method: str,
    query_category: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 120,
) -> Figure:
    """展示单张查询图片及其 Top-5 检索结果

    包含：（1）当前使用的特征方法；（2）查询图片及其类别；
    （3）排名前 5 的检索结果；（4）每张结果图片的排名、类别和距离或相似度
    """
    query_path = Path(query_path)
    top_results = list(results)[:5]
    resolved_query_cat = (
        query_category if query_category is not None else query_path.parent.name
    )

    image_count = len(top_results) + 1
    figure, axes = plt.subplots(
        1,
        image_count,
        figsize=(3.1 * image_count, 4.1),
        squeeze=False,
    )
    row_axes = axes[0]

    query_axis: Axes = row_axes[0]
    query_axis.imshow(_read_canvas_image(query_path))
    query_axis.set_title(
        f"【查询图片】\n类别：{resolved_query_cat}\n{query_path.name}",
        fontsize=9.5,
        color="#1565c0",
        pad=8,
    )
    _style_border(query_axis, "#1565c0", linewidth=2.5)

    for axis, result in zip(row_axes[1:], top_results):
        result_path = Path(result.path)
        axis.imshow(_read_canvas_image(result_path))
        metric_label = METRIC_LABELS[result.metric]
        is_hit = result.category == resolved_query_cat
        border_color = "#2e7d32" if is_hit else "#c62828"
        status_tag = "同类" if is_hit else "异类"

        axis.set_title(
            f"Top-{result.rank} [{status_tag}]\n类别：{result.category}\n"
            f"{metric_label}：{result.score:.4f}\n{result_path.name}",
            fontsize=9.5,
            color=border_color,
            pad=8,
        )
        _style_border(axis, border_color, linewidth=2.0)

    figure.suptitle(
        f"以图搜图检索结果｜特征方法：{feature_method}",
        fontsize=13,
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_method_comparison(
    query_path: str | Path,
    hist_results: Sequence[SearchResult],
    deep_results: Sequence[SearchResult],
    query_category: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 135,
) -> Figure:
    """对同一张查询图上下并排展示两种特征提取方法的 Top-5 检索对比"""
    query_path = Path(query_path)
    resolved_query_cat = (
        query_category if query_category is not None else query_path.parent.name
    )
    rows = [
        ("RGB 颜色直方图 (48 维)", list(hist_results)[:5]),
        ("ResNet-50 深度特征 (2048 维)", list(deep_results)[:5]),
    ]

    figure, axes = plt.subplots(2, 6, figsize=(18.2, 9.2), squeeze=False)
    for row_idx, (method_label, row_results) in enumerate(rows):
        q_ax: Axes = axes[row_idx, 0]
        q_ax.imshow(_read_canvas_image(query_path))
        q_ax.set_title(
            f"【{method_label}】\n查询图｜类别：{resolved_query_cat}\n{query_path.name}",
            fontsize=9.5,
            color="#1565c0",
            pad=7,
        )
        _style_border(q_ax, "#1565c0", linewidth=2.5)

        for col_idx, result in enumerate(row_results, start=1):
            ax: Axes = axes[row_idx, col_idx]
            r_path = Path(result.path)
            ax.imshow(_read_canvas_image(r_path))
            is_hit = result.category == resolved_query_cat
            color = "#2e7d32" if is_hit else "#c62828"
            tag = "同类" if is_hit else "异类"
            metric_label = METRIC_LABELS[result.metric]
            ax.set_title(
                f"Top-{result.rank} [{tag}]\n类别：{result.category}\n"
                f"{metric_label}：{result.score:.4f}\n{r_path.name}",
                fontsize=9.0,
                color=color,
                pad=7,
            )
            _style_border(ax, color, linewidth=2.0)

    figure.suptitle(
        f"两种特征提取方法检索对比｜查询类别：{resolved_query_cat} ({query_path.name})",
        fontsize=13.5,
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.subplots_adjust(hspace=0.42, wspace=0.12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure
