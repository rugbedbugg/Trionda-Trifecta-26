"""Official FIFA group-stage ranking rules.

FIFA ranks a group by, in strict order:

  1. points (3 win / 1 draw)
  2. goal difference across the whole group
  3. goals scored across the whole group

If two or more teams are still equal, the same three criteria are re-applied to
**only the matches between the tied teams** (head-to-head):

  4. head-to-head points
  5. head-to-head goal difference
  6. head-to-head goals scored

and finally fair-play points, then a drawing of lots. We cannot predict
disciplinary cards, so those last two are replaced by a deterministic seed
(pre-tournament Elo, then name) — clearly a substitute for the coin-flip tail,
not a predictive claim.

The 8 best third-placed teams (24 group qualifiers -> 32-team knockout) are
ranked by the same overall criteria 1-3, since head-to-head is undefined across
groups.
"""
from __future__ import annotations

Match = tuple  # (home_key, away_key, home_goals, away_goals)


def compute_records(team_keys, matches) -> dict:
    """Standard P/W/D/L/GF/GA/GD/Pts table over the given teams and matches."""
    rec = {
        t: {"team": t, "p": 0, "w": 0, "d": 0, "l": 0,
            "gf": 0, "ga": 0, "gd": 0, "pts": 0}
        for t in team_keys
    }
    for h, a, hg, ag in matches:
        if h not in rec or a not in rec:
            continue
        rec[h]["p"] += 1
        rec[a]["p"] += 1
        rec[h]["gf"] += hg
        rec[h]["ga"] += ag
        rec[a]["gf"] += ag
        rec[a]["ga"] += hg
        if hg > ag:
            rec[h]["w"] += 1
            rec[a]["l"] += 1
            rec[h]["pts"] += 3
        elif hg < ag:
            rec[a]["w"] += 1
            rec[h]["l"] += 1
            rec[a]["pts"] += 3
        else:
            rec[h]["d"] += 1
            rec[a]["d"] += 1
            rec[h]["pts"] += 1
            rec[a]["pts"] += 1
    for t in rec:
        rec[t]["gd"] = rec[t]["gf"] - rec[t]["ga"]
    return rec


def _overall_key(rec, t, use_goals):
    return (rec[t]["pts"], rec[t]["gd"], rec[t]["gf"]) if use_goals else (rec[t]["pts"],)


def _break_ties(tied, matches, seed_key, use_goals):
    """Order teams that are level on overall points, using head-to-head then the
    deterministic seed (Elo) then name. Goal-based criteria are used only when
    ``use_goals`` is set (i.e. when real scores are available)."""
    h2h = compute_records(tied, [m for m in matches if m[0] in tied and m[1] in tied])

    def sort_key(t):
        goals = (-h2h[t]["gd"], -h2h[t]["gf"]) if use_goals else ()
        return (
            -h2h[t]["pts"], *goals,
            -float(seed_key.get(t, 0.0)) if seed_key else 0.0,
            str(t),
        )

    return sorted(tied, key=sort_key)


def rank_group(team_keys, matches, seed_key=None, use_goals=True):
    """Return (ordered_team_keys, records) for one group.

    With ``use_goals`` (real scores in hand) this is the full official FIFA order
    — points, goal difference, goals scored, then head-to-head. When the model
    predicts only outcomes (no scores), pass ``use_goals=False`` to rank by
    points, then head-to-head points, then team rating.
    """
    rec = compute_records(team_keys, matches)
    order = sorted(team_keys, key=lambda t: _overall_key(rec, t, use_goals), reverse=True)

    resolved = []
    i = 0
    while i < len(order):
        j = i
        while (j + 1 < len(order)
               and _overall_key(rec, order[j + 1], use_goals) == _overall_key(rec, order[i], use_goals)):
            j += 1
        block = order[i:j + 1]
        resolved.extend(
            _break_ties(block, matches, seed_key, use_goals) if len(block) > 1 else block)
        i = j + 1
    return resolved, rec


def rank_thirds(third_entries, seed_key=None, use_goals=True):
    """Rank the third-placed teams across groups; caller takes the top N.

    third_entries: list of dicts with keys team, group, pts (and gd, gf when
    goals are available). Returns the same dicts, best first.
    """
    def sort_key(e):
        goals = (-e["gd"], -e["gf"]) if use_goals else ()
        return (
            -e["pts"], *goals,
            -float(seed_key.get(e["team"], 0.0)) if seed_key else 0.0,
            str(e["team"]),
        )

    return sorted(third_entries, key=sort_key)
