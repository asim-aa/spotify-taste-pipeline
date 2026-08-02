"""Build a single clean tracks dataframe from cached/fetched Spotify data."""

from datetime import datetime, timezone

import pandas as pd

from spotify_client import (
    DATA_DIR,
    fetch_audio_features,
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
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "added_at",
    "days_since_added",
]


def _valid_track_ids(raw_tracks: list) -> list:
    """IDs of tracks that are actual Spotify tracks (not local files, not missing an id)."""
    return [
        item["track"]["id"]
        for item in raw_tracks
        if item.get("track") and not item["track"].get("is_local") and item["track"].get("id")
    ]


def build_dataframe(raw_tracks: list, audio_features: dict) -> tuple[pd.DataFrame, int]:
    """Combine saved-track metadata with audio features into one dataframe.

    Tracks with no usable ID (local files, podcasts) or missing audio features are
    dropped rather than crashing; the drop count is returned alongside the dataframe.
    """
    now = datetime.now(timezone.utc)
    rows = []
    dropped = 0

    for item in raw_tracks:
        track = item.get("track")
        if not track or track.get("is_local") or not track.get("id"):
            dropped += 1
            continue

        features = audio_features.get(track["id"])
        if not features:
            dropped += 1
            continue

        added_at_raw = item.get("added_at")
        added_at = (
            datetime.fromisoformat(added_at_raw.replace("Z", "+00:00")) if added_at_raw else None
        )
        days_since_added = (now - added_at).days if added_at else None

        album = track.get("album") or {}
        release_date = album.get("release_date") or ""
        release_year = int(release_date[:4]) if release_date[:4].isdigit() else None

        rows.append(
            {
                "track_id": track["id"],
                "track_name": track.get("name"),
                "artist_name": ", ".join(a["name"] for a in track.get("artists", [])),
                "album_name": album.get("name"),
                "release_year": release_year,
                "popularity": track.get("popularity"),
                "danceability": features.get("danceability"),
                "energy": features.get("energy"),
                "valence": features.get("valence"),
                "tempo": features.get("tempo"),
                "acousticness": features.get("acousticness"),
                "instrumentalness": features.get("instrumentalness"),
                "added_at": added_at.isoformat() if added_at else None,
                "days_since_added": days_since_added,
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df, dropped


def run_pipeline(force_refresh: bool = False) -> pd.DataFrame:
    sp = get_spotify_client()

    raw_tracks = fetch_saved_tracks(sp, force_refresh=force_refresh)
    track_ids = _valid_track_ids(raw_tracks)
    audio_features = fetch_audio_features(sp, track_ids, force_refresh=force_refresh)

    df, dropped = build_dataframe(raw_tracks, audio_features)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRACKS_CSV_PATH, index=False)

    print("=" * 50)
    print("Spotify pipeline summary")
    print(f"  Total saved tracks pulled:            {len(raw_tracks)}")
    print(f"  Tracks with complete audio features:  {len(df)}")
    print(f"  Tracks dropped (local/no id/no feats): {dropped}")
    print(f"  Saved to: {TRACKS_CSV_PATH}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_pipeline()
