"""Mockup script: look up song info on MusicBrainz given an artist/title.

The input artist/title need not match exactly (typos, alternate spellings,
"feat." variants, etc). We search MusicBrainz with a loose free-text query
and then re-rank the candidates it returns using local string similarity
combined with MusicBrainz's own relevance score.

MusicBrainz requires a descriptive User-Agent with contact info for API
access (see https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting).
Set it via the MB_CONTACT environment variable (an email or project URL).

Usage:
    export MB_CONTACT="you@example.com"
    python fetch_musicbrainz_info.py --artist "Daft Punk" --title "One More Time"
    python fetch_musicbrainz_info.py --csv data/songs.csv --out data/songs_mb.csv --with-tags
"""

import argparse
import os
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

API_URL = "https://musicbrainz.org/ws/2/recording/"
APP_NAME = "misongyny-dataset-paper"
APP_VERSION = "0.1"
RATE_LIMIT_SECONDS = 1.0  # MusicBrainz asks for at most 1 request/second


def _mb_get(url: str, params: dict, contact: str) -> dict:
    headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION} ( {contact} )"}
    response = requests.get(url, params={**params, "fmt": "json"}, headers=headers, timeout=10)
    response.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    return response.json()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()


_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def _lucene_escape(text: str) -> str:
    return "".join(f"\\{ch}" if ch in _LUCENE_SPECIAL else ch for ch in text)


def search_recordings(artist: str, title: str, contact: str, limit: int = 5) -> list[dict]:
    """Field-scoped, unquoted search: keeps title/artist terms in their own fields (so an
    original recording isn't outranked by a cover whose title just mentions the artist)
    while still tolerating individual word differences, since terms aren't matched as an
    exact phrase."""
    query = f"artist:({_lucene_escape(artist)}) AND recording:({_lucene_escape(title)})"
    payload = _mb_get(API_URL, {"query": query, "limit": limit}, contact)

    candidates = []
    for rec in payload.get("recordings", []):
        artist_credit = ", ".join(c.get("name", "") for c in rec.get("artist-credit", []))
        release = (rec.get("releases") or [{}])[0]
        candidates.append(
            {
                "mbid": rec.get("id"),
                "title": rec.get("title"),
                "artist": artist_credit,
                "mb_score": rec.get("score"),
                "length_ms": rec.get("length"),
                "release": release.get("title"),
                "release_date": release.get("date"),
            }
        )
    return candidates


def get_recording_tags(mbid: str, contact: str, limit: int = 5) -> list[str]:
    payload = _mb_get(f"{API_URL}{mbid}", {"inc": "tags"}, contact)
    tags = sorted(payload.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
    return [tag["name"] for tag in tags[:limit]]


def best_match(artist: str, title: str, contact: str, limit: int = 5) -> dict | None:
    """Combine MusicBrainz's relevance score with local string similarity, since input
    spelling/formatting may differ from the canonical MusicBrainz entry."""
    candidates = search_recordings(artist, title, contact, limit=limit)
    if not candidates:
        return None

    for c in candidates:
        title_sim = _similarity(c["title"], title)
        artist_sim = _similarity(c["artist"], artist)
        c["match_score"] = 0.5 * (c["mb_score"] or 0) / 100 + 0.25 * title_sim + 0.25 * artist_sim

    return max(candidates, key=lambda c: c["match_score"])


def lookup_songs(df: pd.DataFrame, contact: str, limit: int = 5, with_tags: bool = False) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        try:
            match = best_match(row["artist"], row["title"], contact, limit=limit)
            if match and with_tags:
                match["tags"] = ", ".join(get_recording_tags(match["mbid"], contact))
        except requests.RequestException:
            match = None
        results.append(match or {})

    result_df = pd.DataFrame(results).add_prefix("mb_")
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up song info on MusicBrainz")
    parser.add_argument("--artist", help="Artist name (approximate match ok)")
    parser.add_argument("--title", help="Track title (approximate match ok)")
    parser.add_argument("--csv", help="CSV with 'artist' and 'title' columns to look up in bulk")
    parser.add_argument("--out", default="musicbrainz_matches.csv", help="Output CSV path for --csv mode")
    parser.add_argument("--limit", type=int, default=5, help="Number of MusicBrainz candidates to consider")
    parser.add_argument("--with-tags", action="store_true", help="Fetch folksonomy tags for the matched recording")
    parser.add_argument(
        "--contact",
        default=os.environ.get("MB_CONTACT"),
        help="Contact info for the MusicBrainz User-Agent (email or URL); defaults to MB_CONTACT env var",
    )
    args = parser.parse_args()

    if not args.contact:
        raise SystemExit(
            "MusicBrainz requires contact info in the User-Agent header. "
            "Set --contact or the MB_CONTACT environment variable."
        )

    if args.artist and args.title:
        match = best_match(args.artist, args.title, args.contact, limit=args.limit)
        if not match:
            print(f"No match found for {args.artist} - {args.title}")
            return
        print(
            f"{args.artist} - {args.title} -> {match['artist']} - {match['title']} "
            f"(mbid={match['mbid']}, match_score={match['match_score']:.2f})"
        )
        if match.get("release"):
            print(f"Release: {match['release']} ({match.get('release_date', '?')})")
        if args.with_tags:
            tags = get_recording_tags(match["mbid"], args.contact)
            print(f"Tags: {', '.join(tags) or '(no tags found)'}")
        return

    if not args.csv:
        raise SystemExit("Provide --artist/--title for a single lookup, or --csv for bulk lookup.")

    df = pd.read_csv(args.csv)
    result_df = lookup_songs(df, args.contact, limit=args.limit, with_tags=args.with_tags)
    result_df.to_csv(args.out, index=False)
    print(f"Saved {len(result_df)} matches to {args.out}")


if __name__ == "__main__":
    main()
