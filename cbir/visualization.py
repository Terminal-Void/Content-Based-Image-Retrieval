"""以图搜图结果可视化与实验分析绘图工具。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from .core import GalleryEvaluation, SearchResult
from .extractors import HistogramFeatureExtractor, load_rgb_image

CategoryResolver = Callable[[Path], str]

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

METRIC_LABELS = {
    "cosine": "余弦相似度",
    "euclidean": "欧氏距离",
    "l1": "L1 距离",
    "intersection": "直方图相交度",
}


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


def _read_image(
    image_path: str | Path,
    canvas_size: tuple[int, int] | None = (280, 280),
) -> Image.Image:
    """读取图片并可选地等比例居中嵌入统一画布，保证多子图标题严格对齐。"""
    image = load_rgb_image(image_path)
    if canvas_size is not None:
        image = ImageOps.pad(
            image,
            canvas_size,
            method=Image.Resampling.BILINEAR,
            color=(246, 247, 249),
        )
    return image


def _style_border(axis, color: str, linewidth: float = 2.2) -> None:
    """为子图添加指定颜色的边框并隐藏坐标轴刻度。"""
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
    category_resolver: CategoryResolver | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 115,
):
    """展示查询图片和前五名检索结果。

    图中包含：（1）当前使用的特征方法；（2）查询图片及其类别；
    （3）排名前 5 的检索结果；（4）每张结果图片的排名、类别和距离/相似度。
    """
    query_path = Path(query_path)
    top_results = list(results)[:5]
    image_count = len(top_results) + 1
    figure, axes = plt.subplots(
        1,
        image_count,
        figsize=(3.1 * image_count, 4.1),
        squeeze=False,
    )
    axes = axes[0]

    resolved_query_cat = _get_category(
        query_path, query_category, category_resolver
    )
    query_axis = axes[0]
    query_axis.imshow(_read_image(query_path))
    query_axis.set_title(
        f"【查询图片】\n类别：{resolved_query_cat}\n{query_path.name}",
        fontsize=9.5,
        color="#1565c0",
        pad=8,
    )
    _style_border(query_axis, "#1565c0", linewidth=2.5)

    for axis, result in zip(axes[1:], top_results):
        result_path = Path(result.path)
        axis.imshow(_read_image(result_path))
        category = _get_category(
            result_path,
            result.category,
            category_resolver,
        )
        metric_label = METRIC_LABELS.get(result.metric, "相似度")
        is_hit = (
            resolved_query_cat != "未标注" and category == resolved_query_cat
        )
        border_color = "#2e7d32" if is_hit else "#c62828"
        status_tag = "同类" if is_hit else "异类"
        if resolved_query_cat == "未标注":
            border_color = "#546e7a"
            status_tag = ""
        tag_suffix = f" [{status_tag}]" if status_tag else ""
        axis.set_title(
            f"Top-{result.rank}{tag_suffix}\n类别：{category}\n"
            f"{metric_label}：{result.score:.4f}\n{result_path.name}",
            fontsize=9.5,
            color=border_color,
            pad=8,
        )
        _style_border(axis, border_color, linewidth=2.0)

    figure.suptitle(
        f"以图搜图检索结果｜特征方法：{feature_method}",
        fontsize=13,
        fontweight="normal",
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
):
    """对同一张查询图同时展示 RGB 颜色直方图与 ResNet-50 深度特征的 Top-5 检索对比。"""
    query_path = Path(query_path)
    resolved_query_cat = query_category or parent_category(query_path)
    rows = [
        ("RGB 颜色直方图 (48 维)", list(hist_results)[:5]),
        ("ResNet-50 深度特征 (2048 维)", list(deep_results)[:5]),
    ]

    figure, axes = plt.subplots(2, 6, figsize=(18.2, 9.2), squeeze=False)
    for row_idx, (method_label, row_results) in enumerate(rows):
        q_ax = axes[row_idx, 0]
        q_ax.imshow(_read_image(query_path))
        q_ax.set_title(
            f"【{method_label}】\n查询图｜类别：{resolved_query_cat}\n{query_path.name}",
            fontsize=9.5,
            color="#1565c0",
            pad=7,
        )
        _style_border(q_ax, "#1565c0", linewidth=2.5)

        for col_idx, result in enumerate(row_results, start=1):
            ax = axes[row_idx, col_idx]
            r_path = Path(result.path)
            ax.imshow(_read_image(r_path))
            cat = result.category or parent_category(r_path)
            is_hit = cat == resolved_query_cat
            color = "#2e7d32" if is_hit else "#c62828"
            tag = "同类" if is_hit else "异类"
            metric_label = METRIC_LABELS.get(result.metric, "余弦相似度")
            ax.set_title(
                f"Top-{result.rank} [{tag}]\n类别：{cat}\n"
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


def plot_dataset_samples(
    sample_paths: Sequence[str | Path],
    cols: int = 5,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 135,
):
    """展示数据集中各类别的代表性样本图片。"""
    paths = [Path(p) for p in sample_paths]
    rows = (len(paths) + cols - 1) // cols
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.1 * cols, 4.1 * rows),
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for ax, path in zip(flat_axes, paths):
        raw_img = load_rgb_image(path)
        w, h = raw_img.size
        ax.imshow(_read_image(path))
        ax.set_title(
            f"类别：{path.parent.name}\n原始尺寸：{w}×{h}",
            fontsize=9.5,
            pad=7,
        )
        _style_border(ax, "#cfd8dc", linewidth=1.2)

    for ax in flat_axes[len(paths) :]:
        ax.axis("off")

    figure.suptitle("数据集 10 个类别代表样本预览", fontsize=13.5, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.subplots_adjust(hspace=0.32, wspace=0.12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_feature_illustration(
    image_path: str | Path,
    hist_extractor: HistogramFeatureExtractor,
    deep_vector: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 140,
):
    """可视化单张图片的特征提取过程：原图、RGB 三通道归一化直方图与 ResNet-50 深度特征分布。"""
    image_path = Path(image_path)
    channel_hists = hist_extractor.extract_channel_histograms(image_path)
    bins = hist_extractor.bins_per_channel
    x = np.arange(bins)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.35, 1.45]},
    )

    axes[0].imshow(_read_image(image_path))
    axes[0].set_title(
        f"输入图像（RGB 预处理后）\n类别：{image_path.parent.name}\n{image_path.name}",
        fontsize=10,
    )
    _style_border(axes[0], "#1565c0", linewidth=1.8)

    colors = ("#e53935", "#43a047", "#1e88e5")
    labels = ("R 通道", "G 通道", "B 通道")
    width = 0.26
    for idx, (color, label) in enumerate(zip(colors, labels)):
        axes[1].bar(
            x + (idx - 1) * width,
            channel_hists[idx],
            width=width,
            color=color,
            alpha=0.85,
            label=label,
        )
    axes[1].set_title(
        f"RGB 三通道归一化直方图（共 3×{bins}={3*bins} 维）",
        fontsize=10,
    )
    axes[1].set_xlabel("颜色亮度区间 (Bin 0 ~ 15)")
    axes[1].set_ylabel("归一化像素占比")
    axes[1].set_xticks(x[::2])
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(
        np.arange(len(deep_vector)),
        deep_vector,
        color="#3949ab",
        linewidth=0.6,
        alpha=0.85,
    )
    axes[2].set_title(
        f"ResNet-50 全局池化深度特征向量（{len(deep_vector)} 维，L2 归一化）",
        fontsize=10,
    )
    axes[2].set_xlabel("特征通道维度索引 (0 ~ 2047)")
    axes[2].set_ylabel("归一化激活值")
    axes[2].set_xlim(0, len(deep_vector))
    axes[2].grid(linestyle="--", alpha=0.35)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_evaluation_comparison(
    hist_eval: GalleryEvaluation,
    deep_eval: GalleryEvaluation,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 140,
):
    """绘制 RGB 颜色直方图与 ResNet-50 深度特征的整体评测指标与分类别 P@5 对比图。"""
    figure, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.5, 4.7),
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )

    metric_names = ["Top-1 准确率", "Hit@5 (前5命中率)", "Precision@5", "mAP@5"]
    hist_vals = [
        hist_eval.top1_accuracy,
        hist_eval.hit_rate_at_k,
        hist_eval.mean_precision_at_k,
        hist_eval.map_at_k,
    ]
    deep_vals = [
        deep_eval.top1_accuracy,
        deep_eval.hit_rate_at_k,
        deep_eval.mean_precision_at_k,
        deep_eval.map_at_k,
    ]

    x1 = np.arange(len(metric_names))
    w1 = 0.34
    bars_h = ax1.bar(
        x1 - w1 / 2,
        [v * 100 for v in hist_vals],
        width=w1,
        color="#fb8c00",
        label=hist_eval.method_name,
    )
    bars_d = ax1.bar(
        x1 + w1 / 2,
        [v * 100 for v in deep_vals],
        width=w1,
        color="#1e88e5",
        label=deep_eval.method_name,
    )
    for bar in list(bars_h) + list(bars_d):
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metric_names, fontsize=9.5)
    ax1.set_ylabel("百分比 (%)")
    ax1.set_ylim(0, 125)
    ax1.set_title("（a）整体检索性能指标对比（20 张查询图）", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=9.5)

    categories = list(hist_eval.category_precision.keys())
    x2 = np.arange(len(categories))
    w2 = 0.36
    cat_h = [hist_eval.category_precision[c] * 100 for c in categories]
    cat_d = [deep_eval.category_precision[c] * 100 for c in categories]
    ax2.bar(
        x2 - w2 / 2,
        cat_h,
        width=w2,
        color="#fb8c00",
        label=hist_eval.method_name,
    )
    ax2.bar(
        x2 + w2 / 2,
        cat_d,
        width=w2,
        color="#1e88e5",
        label=deep_eval.method_name,
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Precision@5 (%)")
    ax2.set_ylim(0, 125)
    ax2.set_title("（b）各语义类别的 Top-5 查准率 (Precision@5) 对比", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=9.5)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure
