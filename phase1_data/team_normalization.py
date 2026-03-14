"""
Team Name Normalization — Map the zoo of NBA team names to canonical IDs.

NBA teams have been called different things across sources and eras:
  - "BOS" vs "Boston Celtics" vs "Celtics"
  - Franchise moves: "Seattle SuperSonics" -> "Oklahoma City Thunder"
  - Name changes: "Washington Bullets" -> "Washington Wizards" (1997)
  - City changes: "New Jersey Nets" -> "Brooklyn Nets"
"""
import re
import json
from pathlib import Path
from difflib import SequenceMatcher


# ── Known aliases (hand-curated for common discrepancies) ──
KNOWN_ALIASES = {
    "Boston Celtics": ["BOS", "Celtics"],
    "Brooklyn Nets": ["BKN", "BRK", "Nets", "New Jersey Nets", "NJN", "NJ Nets"],
    "New York Knicks": ["NYK", "NY Knicks", "Knicks", "Knickerbockers"],
    "Philadelphia 76ers": ["PHI", "76ers", "Sixers", "Philly 76ers",
                           "Syracuse Nationals", "Syracuse Nats"],
    "Toronto Raptors": ["TOR", "Raptors"],
    "Chicago Bulls": ["CHI", "Bulls"],
    "Cleveland Cavaliers": ["CLE", "Cavaliers", "Cavs"],
    "Detroit Pistons": ["DET", "Pistons", "Fort Wayne Pistons"],
    "Indiana Pacers": ["IND", "Pacers"],
    "Milwaukee Bucks": ["MIL", "Bucks"],
    "Atlanta Hawks": ["ATL", "Hawks", "St. Louis Hawks", "Milwaukee Hawks",
                       "Tri-Cities Blackhawks"],
    "Charlotte Hornets": ["CHA", "CHH", "Hornets", "Charlotte Bobcats", "Bobcats"],
    "Miami Heat": ["MIA", "Heat"],
    "Orlando Magic": ["ORL", "Magic"],
    "Washington Wizards": ["WAS", "WSH", "Wizards", "Washington Bullets", "Bullets",
                           "Baltimore Bullets", "Capital Bullets",
                           "Chicago Zephyrs", "Chicago Packers"],
    "Denver Nuggets": ["DEN", "Nuggets"],
    "Minnesota Timberwolves": ["MIN", "Timberwolves", "T-Wolves", "Wolves"],
    "Oklahoma City Thunder": ["OKC", "Thunder", "Seattle SuperSonics",
                               "Seattle Supersonics", "SuperSonics", "Sonics"],
    "Portland Trail Blazers": ["POR", "Trail Blazers", "Blazers"],
    "Utah Jazz": ["UTA", "Jazz", "New Orleans Jazz"],
    "Golden State Warriors": ["GSW", "GS Warriors", "Warriors",
                               "San Francisco Warriors", "Philadelphia Warriors"],
    "Los Angeles Clippers": ["LAC", "LA Clippers", "Clippers",
                              "San Diego Clippers", "Buffalo Braves"],
    "Los Angeles Lakers": ["LAL", "LA Lakers", "Lakers",
                            "Minneapolis Lakers"],
    "Phoenix Suns": ["PHX", "Suns"],
    "Sacramento Kings": ["SAC", "Kings", "Kansas City Kings",
                          "Cincinnati Royals", "Rochester Royals",
                          "Kansas City-Omaha Kings"],
    "Dallas Mavericks": ["DAL", "Mavericks", "Mavs"],
    "Houston Rockets": ["HOU", "Rockets", "San Diego Rockets"],
    "Memphis Grizzlies": ["MEM", "Grizzlies", "Vancouver Grizzlies"],
    "New Orleans Pelicans": ["NOP", "NO Pelicans", "Pelicans",
                              "New Orleans Hornets", "NO Hornets",
                              "New Orleans/Oklahoma City Hornets",
                              "Charlotte Hornets (original)"],
    "San Antonio Spurs": ["SAS", "SA Spurs", "Spurs", "Dallas Chaparrals"],
}

# Franchise relocations (old_name -> current canonical)
FRANCHISE_MOVES = {
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "Seattle Supersonics": "Oklahoma City Thunder",
    "New Jersey Nets": "Brooklyn Nets",
    "Washington Bullets": "Washington Wizards",
    "Baltimore Bullets": "Washington Wizards",
    "Capital Bullets": "Washington Wizards",
    "Chicago Zephyrs": "Washington Wizards",
    "Chicago Packers": "Washington Wizards",
    "Vancouver Grizzlies": "Memphis Grizzlies",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Charlotte Bobcats": "Charlotte Hornets",
    "San Diego Clippers": "Los Angeles Clippers",
    "Buffalo Braves": "Los Angeles Clippers",
    "San Diego Rockets": "Houston Rockets",
    "Kansas City Kings": "Sacramento Kings",
    "Cincinnati Royals": "Sacramento Kings",
    "Rochester Royals": "Sacramento Kings",
    "Minneapolis Lakers": "Los Angeles Lakers",
    "San Francisco Warriors": "Golden State Warriors",
    "Philadelphia Warriors": "Golden State Warriors",
    "Syracuse Nationals": "Philadelphia 76ers",
    "Fort Wayne Pistons": "Detroit Pistons",
    "St. Louis Hawks": "Atlanta Hawks",
    "Milwaukee Hawks": "Atlanta Hawks",
    "Tri-Cities Blackhawks": "Atlanta Hawks",
    "New Orleans Jazz": "Utah Jazz",
    "Dallas Chaparrals": "San Antonio Spurs",
    "Kansas City-Omaha Kings": "Sacramento Kings",
}

STRIP_PATTERNS = [
    r"\s+Univ\.?$", r"\s+University$", r"^University of\s+",
    r"\s+College$", r"^The\s+",
]


class TeamNormalizer:
    """Resolves team names across different data sources to canonical IDs."""

    def __init__(self):
        self.canonical_to_id: dict[str, int] = {}
        self.alias_to_canonical: dict[str, str] = {}
        self.unresolved: list[dict] = []
        self._next_id = 1

        # Load franchise moves first
        for old_name, current in FRANCHISE_MOVES.items():
            self.alias_to_canonical[old_name.lower()] = current
            self.alias_to_canonical[self._clean_name(old_name)] = current

        # Load known aliases
        for canonical, aliases in KNOWN_ALIASES.items():
            self._register(canonical, aliases)

    def _register(self, canonical: str, aliases: list[str] = None):
        clean = self._clean_name(canonical)
        if canonical not in self.canonical_to_id:
            self.canonical_to_id[canonical] = self._next_id
            self._next_id += 1
        self.alias_to_canonical[clean] = canonical
        self.alias_to_canonical[canonical.lower()] = canonical
        if aliases:
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical
                self.alias_to_canonical[self._clean_name(alias)] = canonical

    def _clean_name(self, name: str) -> str:
        name = name.strip()
        for pattern in STRIP_PATTERNS:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip()
        return name.lower()

    def resolve(self, name: str, source: str = "unknown") -> tuple[str, int]:
        if not name:
            return ("Unknown", 0)
        clean = self._clean_name(name)
        if clean in self.alias_to_canonical:
            canonical = self.alias_to_canonical[clean]
            return (canonical, self.canonical_to_id[canonical])
        if name.lower() in self.alias_to_canonical:
            canonical = self.alias_to_canonical[name.lower()]
            return (canonical, self.canonical_to_id[canonical])
        best_match, best_score = self._fuzzy_match(name)
        if best_score >= 0.85:
            self.alias_to_canonical[clean] = best_match
            return (best_match, self.canonical_to_id[best_match])
        self._register(name)
        self.unresolved.append({
            "original_name": name, "source": source,
            "best_fuzzy_match": best_match,
            "fuzzy_score": round(best_score, 3) if best_match else 0,
        })
        return (name, self.canonical_to_id[name])

    def _fuzzy_match(self, name: str) -> tuple[str | None, float]:
        clean = self._clean_name(name)
        best_match, best_score = None, 0.0
        for canonical in self.canonical_to_id:
            score = SequenceMatcher(None, clean, self._clean_name(canonical)).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical
        return (best_match, best_score)

    def get_unresolved_report(self) -> str:
        if not self.unresolved:
            return "All team names resolved successfully!"
        lines = [f"UNRESOLVED TEAM NAMES ({len(self.unresolved)} total):", ""]
        for entry in self.unresolved:
            match_info = ""
            if entry["best_fuzzy_match"]:
                match_info = f" (closest: {entry['best_fuzzy_match']} @ {entry['fuzzy_score']:.0%})"
            lines.append(f"  [{entry['source']}] {entry['original_name']}{match_info}")
        return "\n".join(lines)

    def save(self, path: str):
        data = {
            "canonical_to_id": self.canonical_to_id,
            "alias_to_canonical": self.alias_to_canonical,
            "unresolved": self.unresolved,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.canonical_to_id = data["canonical_to_id"]
        self.alias_to_canonical = data["alias_to_canonical"]
        self.unresolved = data.get("unresolved", [])
        self._next_id = max(int(v) for v in self.canonical_to_id.values()) + 1 if self.canonical_to_id else 1


if __name__ == "__main__":
    norm = TeamNormalizer()
    test_names = [
        "Boston Celtics", "BOS", "Celtics", "Los Angeles Lakers",
        "Seattle SuperSonics", "Oklahoma City Thunder", "Washington Bullets",
        "Washington Wizards", "New Jersey Nets", "Brooklyn Nets",
        "Charlotte Bobcats", "Charlotte Hornets", "Some Random Team",
    ]
    print("Team Name Resolution Tests:")
    for name in test_names:
        canonical, tid = norm.resolve(name, source="test")
        print(f"  {name:40s} -> {canonical} (id={tid})")
    print(f"\n{norm.get_unresolved_report()}")
