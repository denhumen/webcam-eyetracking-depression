"""
Figure-saving utility for the visualization layer.
"""

from __future__ import annotations
import re
from pathlib import Path
import matplotlib.pyplot as plt
import config


_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitise(name: str) -> str:
    """
    Make a string safe for use as a filename stem on common filesystems
    """
    cleaned = _NAME_SAFE_RE.sub("_", name).strip("_")
    if not cleaned:
        raise ValueError(f"Figure name is empty after sanitisation: {name!r}")
    return cleaned


def save_figure(
    fig: plt.Figure,
    name: str,
    subfolder: str | None = None,
    formats: tuple[str, ...] = ("png"),
    dpi: int = 200,
    close: bool = False,
) -> list[Path]:
    """
    Save a matplotlib figure under config.REPORTS_DIR
    """
    if subfolder:
        out_dir = config.REPORTS_DIR / subfolder
    else:
        out_dir = config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitise(name)
    written: list[Path] = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)

    if close:
        plt.close(fig)

    return written
