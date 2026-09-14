"""Mockup script: fetch songs from the Last.fm API.

Get a free API key at https://www.last.fm/api/account/create
and set it as the LASTFM_API_KEY environment variable.

Usage:
    export LASTFM_API_KEY=your_key_here
    python fetch_lastfm_songs.py --tag hip-hop --limit 50 --out data/lastfm_tracks.csv
    python fetch_lastfm_songs.py --artist "Daft Punk" --title "One More Time"
"""

import argparse
import os
import time

import pandas as pd
import requests

API_URL = "https://ws.audioscrobbler.com/2.0/"


def get_top_tracks_by_tag(tag: str, api_key: str, limit: int = 50, pages: int = 1) -> list[dict]:
    tracks = []
    for page in range(1, pages + 1):
        params = {
            "method": "tag.gettoptracks",
            "tag": tag,
            "api_key": api_key,
            "format": "json",
            "limit": limit,
            "page": page,
        }
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        for track in payload.get("tracks", {}).get("track", []):
            tracks.append(
                {
                    "title": track.get("name"),
                    "artist": track.get("artist", {}).get("name"),
                    "url": track.get("url"),
                    "listeners": track.get("listeners"),
                }
            )
        time.sleep(0.25)  # be nice to the API

    return tracks


def get_track_genres(artist: str, title: str, api_key: str, limit: int = 5) -> list[str]:
    """Last.fm has no dedicated genre field; user-applied tags are the standard proxy."""
    params = {
        "method": "track.gettoptags",
        "artist": artist,
        "track": title,
        "api_key": api_key,
        "format": "json",
    }
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    tags = payload.get("toptags", {}).get("tag", [])
    return [tag["name"] for tag in tags[:limit]]


def add_genres(df: pd.DataFrame, api_key: str, limit: int = 5) -> pd.DataFrame:
    genres = []
    for _, row in df.iterrows():
        try:
            genres.append(", ".join(get_track_genres(row["artist"], row["title"], api_key, limit=limit)))
        except requests.RequestException:
            genres.append("")
        time.sleep(0.25)  # be nice to the API
    df["genres"] = genres
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch songs / genres from Last.fm")
    parser.add_argument("--tag", help="Last.fm tag to fetch top tracks for, e.g. 'hip-hop'")
    parser.add_argument("--limit", type=int, default=50, help="Tracks per page / genres per track")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to fetch")
    parser.add_argument("--out", default="lastfm_tracks.csv", help="Output CSV path")
    parser.add_argument("--artist", help="Look up genres for a single song: artist name")
    parser.add_argument("--title", help="Look up genres for a single song: track title")
    parser.add_argument("--with-genres", action="store_true", help="Add a genres column when fetching by --tag")
    args = parser.parse_args()

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        raise SystemExit("Set the LASTFM_API_KEY environment variable first.")

    if args.artist and args.title:
        genres = get_track_genres(args.artist, args.title, api_key, limit=args.limit)
        print(f"{args.artist} - {args.title}: {', '.join(genres) or '(no tags found)'}")
        return

    if not args.tag:
        raise SystemExit("Provide --tag to fetch tracks, or --artist/--title to look up genres.")

    tracks = get_top_tracks_by_tag(args.tag, api_key, limit=args.limit, pages=args.pages)
    df = pd.DataFrame(tracks)
    if args.with_genres:
        df = add_genres(df, api_key)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} tracks to {args.out}")


if __name__ == "__main__":
    main()
