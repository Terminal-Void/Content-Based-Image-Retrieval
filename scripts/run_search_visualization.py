"""运行一次以图搜图检索并保存查询图与 Top-5 结果图。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cbir import (
    DeepFeatureExtractor,
    FeatureExtractor,
    FeatureGallery,
    HistogramFeatureExtractor,
    list_image_paths,
    plot_search_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        choices=("histogram", "deep"),
        default="histogram",
        help="使用的特征提取方法",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="使用的相似度或距离度量方式",
    )
    parser.add_argument("--query", type=Path, help="查询图片路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/search_result.png"),
        help="结果图保存路径",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset"),
        help="包含 query/ 和 gallery/ 子目录的数据集路径",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不打开图形窗口",
    )
    args = parser.parse_args()

    gallery_paths = list_image_paths(args.data_root / "gallery")
    if not gallery_paths:
        raise FileNotFoundError(
            "未找到检索库图片，请检查 dataset/gallery 下是否按类别分目录。"
        )

    query_path: Path | None = args.query
    if query_path is None:
        query_candidates = list_image_paths(args.data_root / "query")
        if not query_candidates:
            raise FileNotFoundError(
                "未找到查询集图片，请检查 dataset/query 下是否按类别分目录。"
            )
        query_path = query_candidates[0]
    if not query_path.exists():
        raise FileNotFoundError(f"查询图片不存在：{query_path}")

    extractor: FeatureExtractor
    if args.feature == "histogram":
        extractor = HistogramFeatureExtractor(bins_per_channel=16)
        feature_method = "RGB 颜色直方图"
        cache_path = Path("cache/histogram_gallery.npz")
    else:
        extractor = DeepFeatureExtractor()
        feature_method = "ResNet-50 深度特征"
        cache_path = Path("cache/deep_gallery.npz")

    gallery = FeatureGallery.build_or_load(
        gallery_paths,
        extractor,
        cache_path=cache_path,
    )
    query_feature = extractor.extract(query_path)
    results = gallery.search(query_feature, top_k=5, metric=args.metric)
    plot_search_results(
        query_path=query_path,
        results=results,
        feature_method=feature_method,
        save_path=args.output,
        show=not args.no_show,
    )
    print(f"结果图已保存至：{args.output}")


if __name__ == "__main__":
    main()
