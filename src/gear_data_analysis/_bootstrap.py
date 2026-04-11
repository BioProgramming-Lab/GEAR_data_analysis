from __future__ import annotations

import sys
from pathlib import Path

from ._paths import REPO_ROOT


def _looks_like_repo_root(candidate: Path) -> bool:
    return (candidate / "pyproject.toml").exists() and (
        candidate / "src" / "gear_data_analysis" / "__init__.py"
    ).exists()


def find_repo_root(start: Path | None = None) -> Path:
    if start is not None:
        start = Path(start).resolve()
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists() or _looks_like_repo_root(candidate):
                return candidate

    if _looks_like_repo_root(REPO_ROOT):
        return REPO_ROOT

    raise FileNotFoundError("Repository root not found from current working directory.")


def ensure_src_on_path(start: Path | None = None) -> tuple[Path, Path]:
    repo_root = find_repo_root(start)
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root, src_root


def bootstrap_notebook(start: Path | None = None) -> tuple[Path, Path]:
    return ensure_src_on_path(start)


__all__ = ["bootstrap_notebook", "ensure_src_on_path", "find_repo_root"]
