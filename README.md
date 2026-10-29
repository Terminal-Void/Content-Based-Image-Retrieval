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

## 当前库接口

特征提取器、特征对象和特征库位于 `cbir` 包中：

```python
from pathlib import Path

from cbir import FeatureGallery, HistogramFeatureExtractor

extractor = HistogramFeatureExtractor(bins_per_channel=16)
gallery_paths = sorted(Path("dataset/gallery").glob("*.png"))
gallery = FeatureGallery.build_or_load(
    gallery_paths,
    extractor,
    cache_path="cache/rgb_histogram.npz",
)

query = extractor.extract("dataset/query/query_01.png")
results = gallery.search(query, top_k=5)

for result in results:
    print(result.rank, result.score, result.path)
```

`HistogramFeatureExtractor` 在 RGB 颜色空间中分别统计三个通道的直方图，将每个通道归一化后拼接，并进行 L2 归一化。由于特征向量已经归一化，`FeatureGallery.search()` 使用点积计算余弦相似度。

## 结果可视化

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
