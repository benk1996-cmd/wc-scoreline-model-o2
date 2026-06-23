"""Team-name normalisation -- the single most error-prone part of joining
football data. Each source spells countries differently ("USA" / "United States"
/ "United States of America"; "Korea Republic" / "South Korea"). Every name from
every source goes through `canonical()` before it is used as a join key.

The canonical spelling here follows the martj42 results dataset, which we treat
as the spine. Extend ALIASES whenever you wire in a new source and hit a miss --
`unmatched()` helps you find them.
"""
from __future__ import annotations
import unicodedata

# alias (lower-case, accent-stripped) -> canonical name used as the join key
ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "china pr": "China",
    "czechia": "Czech Republic",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo dr": "DR Congo",
    "cape verde": "Cape Verde Islands",
    "the gambia": "Gambia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "bosnia": "Bosnia and Herzegovina",
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "republic of ireland": "Republic of Ireland",
    "ireland": "Republic of Ireland",
}

_KNOWN: set[str] = set()  # populated by register_known(); used by unmatched()


def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().strip().split())


def canonical(name: str) -> str:
    """Return the canonical team name for any source spelling."""
    if name is None:
        return ""
    k = _key(name)
    if k in ALIASES:
        return ALIASES[k]
    return " ".join(str(name).strip().split())


def register_known(names) -> None:
    """Record the spine's team names so `unmatched()` can flag join gaps."""
    for n in names:
        _KNOWN.add(canonical(n))


def unmatched(names) -> set[str]:
    """Names from another source that don't map onto a known spine team.
    Print this when adding a source, then add the misses to ALIASES."""
    return {canonical(n) for n in names if canonical(n) not in _KNOWN}
