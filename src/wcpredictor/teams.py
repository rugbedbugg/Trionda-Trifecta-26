"""Cross-era team canonicalisation.

Fjelstul (1930-2022) and the 2026 dataset spell the same nation differently
("USA" vs "United States", "Türkiye" vs "Turkey", ...) and FIFA folds some
historical entities into modern ones (West Germany -> Germany). To carry a
team's pedigree and Elo forward from history into 2026 we need one canonical
key per nation. This module is the single source of truth for that mapping.
"""
from __future__ import annotations

import unicodedata

# Normalised-name -> canonical normalised-name. Left side is the *variant*,
# right side is the nation we fold it into. Only genuine identity matches go
# here; defunct entities with no modern successor (Yugoslavia, Soviet Union,
# Czechoslovakia, Dutch East Indies, Zaire) are deliberately left alone.
_ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "czechia": "czech republic",
    "cote d'ivoire": "ivory coast",
    "ir iran": "iran",
    "turkey": "turkiye",
    "korea republic": "south korea",
    "korea, south": "south korea",
    "china pr": "china",
    "west germany": "germany",  # FIFA credits West Germany's record to Germany
    # Kaggle international-results names -> our 2026 dataset spellings
    "cape verde": "cabo verde",
    "dr congo": "congo dr",
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize_team(name: str) -> str:
    """Lower-case, de-accent, collapse whitespace. Not yet aliased."""
    if name is None:
        return ""
    n = _strip_accents(str(name)).lower().strip()
    n = " ".join(n.split())
    return n


def canonical_team(name: str) -> str:
    """The canonical key a nation is joined on across all sources."""
    n = normalize_team(name)
    return _ALIASES.get(n, n)
