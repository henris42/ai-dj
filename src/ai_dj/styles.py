"""Style definitions for steering the live mix.

Each style is a display name plus a set of lowercase substrings we match against
a track's genre tag. After the CLAP tagging pass lands, `match` prefers the
AI-derived `ai_genre` field over the iTunes-supplied `genre` so styles become
accurate even when iTunes tags are missing or junk."""
from __future__ import annotations

STYLES: dict[str, list[str]] = {
    "Rock":       ["rock", "grunge", "punk", "alternative", "indie"],
    "Pop":        ["pop"],
    "Electronic": ["electronic", "electronica", "idm", "braindance", "techno", "house", "trance"],
    "Dance":      ["dance", "disco", "house", "club"],
    "Ambient":    ["ambient", "drone", "lo-fi", "chillout", "downtempo"],
    "Classical":  ["classical", "baroque", "orchestral", "romantic"],
    "Jazz":       ["jazz", "fusion", "bebop", "bossa"],
    "Metal":      ["metal", "doom", "thrash", "death"],
    "Hip-Hop":    ["hip hop", "hip-hop", "rap", "r&b", "rnb"],
    "Soul":       ["soul", "blues", "motown", "funk"],
    "Folk":       ["folk", "country", "americana", "celtic", "world"],
    "Soundtrack": ["soundtrack", "o.s.t", "score", "cinematic"],
}


# Per-genre colors for the map + style buttons. Hand-picked for distinguishability
# on a dark background (#111318); adjacent genres get related but distinct hues.
STYLE_COLORS: dict[str, str] = {
    "Rock":       "#e74c3c",  # red
    "Metal":      "#8e2100",  # dark red / blood
    "Pop":        "#ff6fc0",  # hot pink
    "Hip-Hop":    "#a855f7",  # violet
    "Soul":       "#ff8b3d",  # orange
    "Jazz":       "#f5b400",  # amber
    "Folk":       "#a3d42f",  # lime
    "Classical":  "#21c55d",  # green
    "Ambient":    "#b794ff",  # lavender
    "Electronic": "#22d3ee",  # cyan
    "Dance":      "#fde047",  # yellow
    "Soundtrack": "#9ca3af",  # slate
}
UNTAGGED_COLOR = "#3a3f47"  # very dim for tracks without ai_genre


def color_for_genre(genre: str | None) -> str:
    if genre and genre in STYLE_COLORS:
        return STYLE_COLORS[genre]
    return UNTAGGED_COLOR


def match(payload: dict, active: set[str]) -> bool:
    """True if the track (via its Qdrant payload) matches any active style.

    Preference order for the tag source: `ai_genre` (CLAP-derived, when present)
    beats `genre` (iTunes tag). Both are substring-matched case-insensitively."""
    if not active:
        return True
    ai = (payload.get("ai_genre") or "").lower()
    itu = (payload.get("genre") or "").lower()
    for style in active:
        needles = STYLES.get(style, [])
        if ai and any(n in ai for n in needles):
            return True
        # Only consult iTunes tag if there's no AI tag (AI tag wins the vote).
        if not ai and itu and any(n in itu for n in needles):
            return True
    return False
