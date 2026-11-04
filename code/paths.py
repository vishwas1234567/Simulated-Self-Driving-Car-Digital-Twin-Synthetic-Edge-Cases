"""Single source of truth for output folders (project root = parent of code/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, LOG, BUILD = (ROOT / d for d in ("results", "figures", "logs", "build"))
for _d in (RES, FIG, LOG, BUILD):
    _d.mkdir(exist_ok=True)
