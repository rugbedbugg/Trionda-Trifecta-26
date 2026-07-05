"""Central path configuration.

Everything is resolved relative to the repository layout so the project runs
from a clean checkout with no environment variables. The two raw data sources
live next to ``predictor/`` inside the ``wc26`` directory.
"""
from __future__ import annotations

from pathlib import Path

# predictor/src/wcpredictor/paths.py  ->  predictor/
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# wc26/  (holds the raw source repos alongside predictor/)
WC26_ROOT = PACKAGE_ROOT.parent

# ---- Raw sources (read-only) ------------------------------------------------
FJELSTUL_CSV = WC26_ROOT / "worldcup" / "data-csv"
WC2026_DIR = WC26_ROOT / "FIFA-World-Cup-2026-Dataset"
WC2026_SQLITE = WC2026_DIR / "sqlite_fifa_world_cup_2026.db"

# ---- Generated artifacts (writable) -----------------------------------------
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"

UNIFIED_DB = DATA_DIR / "unified.db"
PLAYER_LINK_OVERRIDES = DATA_DIR / "player_link_overrides.csv"
UNRESOLVED_LOG = OUTPUT_DIR / "unresolved_players.csv"


def ensure_dirs() -> None:
    """Create the writable directories if they do not yet exist."""
    for d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
