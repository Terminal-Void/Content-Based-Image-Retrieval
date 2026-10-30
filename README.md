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

## 库接口示例

特征提取器、特征对象、特征库及评测可视化工具位于 `cbir` 包中：

```python
from pathlib import Path

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
    categories=[p.parent.name for p in gallery_paths],
)

query_path = select_query_paths("dataset/query", per_category=2)[0]
query = extractor.extract(query_path)
results = gallery.search(query, top_k=5, metric="cosine")

for result in results:
    print(result.rank, result.category, f"{result.score:.4f}", result.path)
```

- `HistogramFeatureExtractor`：在 RGB 颜色空间中分别统计三个通道的直方图，将每个通道按像素总数归一化后拼接（48 维），再进行 $L_2$ 归一化。
- `DeepFeatureExtractor`：使用 ImageNet 预训练 `ResNet-50` 全局平均池化层输出的 2048 维特征向量并进行 $L_2$ 归一化。
- `FeatureGallery.search()`：默认使用余弦相似度（点积），同时支持欧氏距离（`euclidean`）、街区距离（`l1`）与直方图相交度（`intersection`）。

## 结果可视化与实验复现

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

完整实验流程、20 张查询图在两种特征下的 Top-5 检索对比及定量评测见 `image_retrieval.ipynb`，实验报告见 `report.typ`（编译产物为 `report.pdf`）。
