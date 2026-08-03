"""Build a single clean tracks dataframe from metadata + behavioral signals.

No audio-features dependency (that endpoint 403s for apps created after Nov 2024),
no preview_url usage (also unavailable to new apps), and no genre/artist lookup
(the /v1/artists endpoint 403s the same way — both are gated behind Extended
Quota Mode approval this app doesn't have). Signal comes from saved tracks,
recently-played history, and playlist co-occurrence.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from spotify_client import (
    DATA_DIR,
    fetch_playlists_with_tracks,
    fetch_recently_played,
    fetch_saved_tracks,
    get_spotify_client,
)

TRACKS_CSV_PATH = DATA_DIR / "tracks.csv"

COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "release_year",
    "popularity",
    "added_at",
    "days_since_added",
    "play_recency_days",
    "playlist_count",
    "artist_track_count",
    "image_url",
]


def _valid_saved_items(raw_tracks: list) -> list:
    """Saved-track entries that are actual Spotify tracks (not local files, have an id)."""
    return [
        item
        for item in raw_tracks
        if item.get("track") and not item["track"].get("is_local") and item["track"].get("id")
    ]


def _primary_artist_id(track: dict) -> str | None:
    artists = track.get("artists") or []
    return artists[0].get("id") if artists else None


def _play_recency_map(recently_played: list) -> dict:
    """track_id -> days since its most recent play in the recently-played history."""
    now = datetime.now(timezone.utc)
    recency = {}
    for item in recently_played:
        track = item.get("track")
        played_at_raw = item.get("played_at")
        if not track or not track.get("id") or not played_at_raw:
            continue
        played_at = datetime.fromisoformat(played_at_raw.replace("Z", "+00:00"))
        days = (now - played_at).days
        tid = track["id"]
        if tid not in recency or days < recency[tid]:
            recency[tid] = days
    return recency


def _playlist_count_map(playlists: list) -> dict:
    """track_id -> number of the user's playlists containing it."""
    counts = {}
    for pl in playlists:
        for tid in pl.get("track_ids", []):
            counts[tid] = counts.get(tid, 0) + 1
    return counts


def build_dataframe(
    raw_tracks: list, recently_played: list, playlists: list
) -> tuple[pd.DataFrame, int]:
    """Combine saved-track metadata with behavioral signals into one dataframe.

    Tracks with no usable ID (local files, podcasts) are dropped rather than
    crashing; the drop count is returned alongside the dataframe.
    """
    valid_items = _valid_saved_items(raw_tracks)
    dropped = len(raw_tracks) - len(valid_items)

    play_recency = _play_recency_map(recently_played)
    playlist_counts = _playlist_count_map(playlists)

    artist_counts = {}
    for item in valid_items:
        aid = _primary_artist_id(item["track"])
        artist_counts[aid] = artist_counts.get(aid, 0) + 1

    now = datetime.now(timezone.utc)
    rows = []

    for item in valid_items:
        track = item["track"]
        tid = track["id"]
        aid = _primary_artist_id(track)

        added_at_raw = item.get("added_at")
        added_at = (
            datetime.fromisoformat(added_at_raw.replace("Z", "+00:00")) if added_at_raw else None
        )
        days_since_added = (now - added_at).days if added_at else None

        album = track.get("album") or {}
        release_date = album.get("release_date") or ""
        release_year = int(release_date[:4]) if release_date[:4].isdigit() else None
        images = album.get("images") or []
        image_url = images[0]["url"] if images else None

        rows.append(
            {
                "track_id": tid,
                "track_name": track.get("name"),
                "artist_name": ", ".join(a["name"] for a in track.get("artists", [])),
                "album_name": album.get("name"),
                "release_year": release_year,
                "popularity": track.get("popularity"),
                "added_at": added_at.isoformat() if added_at else None,
                "days_since_added": days_since_added,
                "play_recency_days": play_recency.get(tid),
                "playlist_count": playlist_counts.get(tid, 0),
                "artist_track_count": artist_counts.get(aid, 1),
                "image_url": image_url,
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df, dropped


def run_pipeline(force_refresh: bool = False) -> pd.DataFrame:
    sp = get_spotify_client()

    raw_tracks = fetch_saved_tracks(sp, force_refresh=force_refresh)
    recently_played = fetch_recently_played(sp, force_refresh=force_refresh)
    playlists = fetch_playlists_with_tracks(sp, force_refresh=force_refresh)

    df, dropped = build_dataframe(raw_tracks, recently_played, playlists)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACKS_CSV_PATH, index=False)

    print("=" * 50)
    print("Spotify pipeline summary")
    print(f"  Total saved tracks pulled:               {len(raw_tracks)}")
    print(f"  Tracks with complete metadata:            {len(df)}")
    print(f"  Tracks dropped (local file / no track id): {dropped}")
    print(f"  Saved to: {TRACKS_CSV_PATH}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_pipeline()
