## 环境管理

项目使用 `uv` 管理 Python 版本和依赖：

```bash
uv sync
uv run python --version
```

激活环境不是必须的；需要交互式使用时可以执行：

```bash
source .venv/bin/activate
```

当前作业数据集使用 `dataset/query` 和 `dataset/gallery` 两个目录，两个目录下再按类别分文件夹。类别由父目录名称提供，不从文件名推断。

## 模块结构与接口示例

核心代码位于 `cbir` 包中，按作业流程划分为四个精简模块：

- `cbir/dataset.py`：扫描数据集目录并按类别抽取查询集（`list_image_paths`、`select_query_paths`）；
- `cbir/extractors.py`：图像统一读取预处理（`load_rgb_image`）、带提取器类型标识的 `Feature`、基于 `__init_subclass__` 自动注册的抽象基类 `FeatureExtractor`，以及 `HistogramFeatureExtractor`（48 维 RGB 归一化直方图）和 `DeepFeatureExtractor`（2048 维预训练 ResNet-50 全局平均池化特征）；
- `cbir/gallery.py`：特征检索库 `FeatureGallery`，支持提取器类型一致性校验、余弦相似度（`cosine`）与欧氏距离（`euclidean`）Top-$K$ 排序及 `.npz` 缓存读写；
- `cbir/visualization.py`：Top-5 检索结果展示（`plot_search_results`）与双方法同图对比（`plot_method_comparison`）。

调用示例：

```python
from cbir import (
    FeatureGallery,
    HistogramFeatureExtractor,
    list_image_paths,
    select_query_paths,
)

extractor = HistogramFeatureExtractor(bins_per_channel=16)
gallery_paths = list_image_paths("dataset/gallery")
gallery = FeatureGallery.build_or_load(
    gallery_paths,
    extractor,
    cache_path="cache/histogram_gallery.npz",
)

query_path = select_query_paths("dataset/query", per_category=2)[0]
query_feature = extractor.extract(query_path)
results = gallery.search(query_feature, top_k=5, metric="cosine")

for r in results:
    print(r.rank, r.category, f"{r.score:.4f}", r.path)
```

## 命令行可视化与实验复现

单张查询检索可视化命令：

```bash
uv run python scripts/run_search_visualization.py \
  --feature histogram \
  --no-show \
  --output results/histogram.png
```

深度特征版本：

```bash
uv run python scripts/run_search_visualization.py \
  --feature deep \
  --no-show \
  --output results/deep.png
```

完整实验流程、20 张查询图在两种特征下的 Top-5 检索对比及定量统计见 `image_retrieval.ipynb`，实验报告见 `report.typ`（编译产物为 `report.pdf`）。
