"""数据集目录扫描与查询集划分工具"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_category_images(category_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(category_dir.iterdir())
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


def list_image_paths(root: str | Path) -> list[Path]:
    """扫描按类别子目录组织的图片路径，忽略隐藏文件与隐藏目录"""
    root_path = Path(root)
    if not root_path.exists():
        return []
    paths: list[Path] = []
    for category_dir in sorted(root_path.iterdir()):
        if category_dir.is_dir() and not category_dir.name.startswith("."):
            paths.extend(_iter_category_images(category_dir))
    return paths


def select_query_paths(query_root: str | Path, per_category: int = 2) -> list[Path]:
    """从每个类别子目录中按顺序选取前 ``per_category`` 张图片作为查询集"""
    if per_category < 1:
        raise ValueError("per_category 必须是正整数")
    root_path = Path(query_root)
    if not root_path.exists():
        return []
    selected: list[Path] = []
    for category_dir in sorted(root_path.iterdir()):
        if category_dir.is_dir() and not category_dir.name.startswith("."):
            selected.extend(_iter_category_images(category_dir)[:per_category])
    return selected

