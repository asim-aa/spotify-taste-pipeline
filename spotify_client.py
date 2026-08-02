"""Spotify API client: OAuth, paginated fetches, rate-limit handling, and local caching."""

import json
import os
import time
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_TRACKS_PATH = DATA_DIR / "raw_tracks.json"
RAW_AUDIO_FEATURES_PATH = DATA_DIR / "raw_audio_features.json"
RAW_RECENTLY_PLAYED_PATH = DATA_DIR / "raw_recently_played.json"
TOKEN_CACHE_PATH = BASE_DIR / ".spotify_token_cache"

SCOPES = "user-library-read user-read-recently-played user-top-read"

SAVED_TRACKS_PAGE_SIZE = 50
AUDIO_FEATURES_BATCH_SIZE = 100


def get_spotify_client() -> spotipy.Spotify:
    """Build an authenticated Spotify client using the Authorization Code flow."""
    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")

    missing = [
        name
        for name, val in (
            ("SPOTIFY_CLIENT_ID", client_id),
            ("SPOTIFY_CLIENT_SECRET", client_secret),
            ("SPOTIFY_REDIRECT_URI", redirect_uri),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Spotify app credentials."
        )

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPES,
        cache_path=str(TOKEN_CACHE_PATH),
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _call_with_retry(func, *args, max_retries=5, **kwargs):
    """Call a spotipy method, retrying on 429s and honoring Retry-After."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429 and attempt < max_retries:
                retry_after = 1
                if e.headers:
                    try:
                        retry_after = int(e.headers.get("Retry-After", 1))
                    except (TypeError, ValueError):
                        retry_after = 1
                print(f"Rate limited (429). Waiting {retry_after}s before retry {attempt}/{max_retries}...")
                time.sleep(retry_after + 1)
                continue
            raise
    raise RuntimeError(f"Exceeded max retries ({max_retries}) due to rate limiting")


def fetch_saved_tracks(sp: spotipy.Spotify, force_refresh: bool = False) -> list:
    """Fetch all of the user's saved tracks, paginating 50 at a time. Cached to disk."""
    if not force_refresh and RAW_TRACKS_PATH.exists():
        print(f"Loading saved tracks from cache ({RAW_TRACKS_PATH})...")
        with open(RAW_TRACKS_PATH) as f:
            return json.load(f)

    print("Fetching saved tracks from Spotify API...")
    items = []
    offset = 0
    while True:
        results = _call_with_retry(
            sp.current_user_saved_tracks, limit=SAVED_TRACKS_PAGE_SIZE, offset=offset
        )
        batch = results.get("items", [])
        if not batch:
            break
        items.extend(batch)
        print(f"  fetched {len(items)} tracks so far...")
        if len(batch) < SAVED_TRACKS_PAGE_SIZE:
            break
        offset += SAVED_TRACKS_PAGE_SIZE

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_TRACKS_PATH, "w") as f:
        json.dump(items, f)
    return items


def fetch_audio_features(sp: spotipy.Spotify, track_ids: list, force_refresh: bool = False) -> dict:
    """Fetch audio features for the given track IDs, batching 100 at a time.

    Cached to disk as {track_id: features_or_None}; only missing IDs are fetched on rerun.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache = {}
    if not force_refresh and RAW_AUDIO_FEATURES_PATH.exists():
        with open(RAW_AUDIO_FEATURES_PATH) as f:
            cache = json.load(f)

    missing_ids = [tid for tid in track_ids if tid not in cache]
    if missing_ids:
        print(f"Fetching audio features for {len(missing_ids)} tracks...")
        for i in range(0, len(missing_ids), AUDIO_FEATURES_BATCH_SIZE):
            batch = missing_ids[i : i + AUDIO_FEATURES_BATCH_SIZE]
            results = _call_with_retry(sp.audio_features, batch)
            for tid, feat in zip(batch, results):
                cache[tid] = feat
            print(f"  fetched features for {min(i + AUDIO_FEATURES_BATCH_SIZE, len(missing_ids))}/{len(missing_ids)}...")
        with open(RAW_AUDIO_FEATURES_PATH, "w") as f:
            json.dump(cache, f)
    else:
        print("All requested audio features already cached.")

    return {tid: cache[tid] for tid in track_ids if tid in cache}


def fetch_recently_played(sp: spotipy.Spotify, limit: int = 50, force_refresh: bool = False) -> list:
    """Fetch recently played tracks (Spotify only exposes the last ~50). Cached to disk."""
    if not force_refresh and RAW_RECENTLY_PLAYED_PATH.exists():
        print(f"Loading recently played from cache ({RAW_RECENTLY_PLAYED_PATH})...")
        with open(RAW_RECENTLY_PLAYED_PATH) as f:
            return json.load(f)

    print("Fetching recently played tracks from Spotify API...")
    results = _call_with_retry(sp.current_user_recently_played, limit=limit)
    items = results.get("items", [])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_RECENTLY_PLAYED_PATH, "w") as f:
        json.dump(items, f)
    return items
