"""Spotify API client: OAuth, paginated fetches, rate-limit handling, and local caching.

Only metadata/behavioral endpoints are used here (saved tracks, recently played,
playlists). The audio-features and artists (genre) endpoints are not used — both
403 for this app (Development Mode apps don't get access to those catalog
endpoints without Extended Quota Mode approval).
"""

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
RAW_RECENTLY_PLAYED_PATH = DATA_DIR / "raw_recently_played.json"
RAW_PLAYLISTS_PATH = DATA_DIR / "raw_playlists.json"
RAW_ALBUMS_PATH = DATA_DIR / "raw_albums.json"
TOKEN_CACHE_PATH = BASE_DIR / ".spotify_token_cache"

SCOPES = (
    "user-library-read user-library-modify user-read-recently-played user-top-read "
    "playlist-read-private playlist-read-collaborative "
    "playlist-modify-public playlist-modify-private"
)

SAVED_TRACKS_PAGE_SIZE = 50
PLAYLISTS_PAGE_SIZE = 50
PLAYLIST_ITEMS_PAGE_SIZE = 100
SAVED_ALBUMS_PAGE_SIZE = 50


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


def _fetch_playlist_track_ids(sp: spotipy.Spotify, playlist_id: str) -> list:
    """Paginate a single playlist's items (100 per call) and return its track IDs.

    Each entry nests its content under "item" (not the older "track" key), with a
    "type" field ("track" vs "episode") to distinguish tracks from podcast episodes.
    """
    track_ids = []
    offset = 0
    while True:
        results = _call_with_retry(
            sp.playlist_items,
            playlist_id,
            limit=PLAYLIST_ITEMS_PAGE_SIZE,
            offset=offset,
            fields="items(item(id,type)),total",
            additional_types=["track"],
        )
        batch = results.get("items", [])
        if not batch:
            break
        for entry in batch:
            item = entry.get("item")
            if item and item.get("type") == "track" and item.get("id"):
                track_ids.append(item["id"])
        if len(batch) < PLAYLIST_ITEMS_PAGE_SIZE:
            break
        offset += PLAYLIST_ITEMS_PAGE_SIZE
    return track_ids


def fetch_playlists_with_tracks(sp: spotipy.Spotify, force_refresh: bool = False) -> list:
    """Fetch all of the user's playlists and the track IDs each contains.

    Returns a list of {"id", "name", "image_url", "is_owned", "track_ids"} dicts.
    Cached to disk.
    """
    if not force_refresh and RAW_PLAYLISTS_PATH.exists():
        print(f"Loading playlists from cache ({RAW_PLAYLISTS_PATH})...")
        with open(RAW_PLAYLISTS_PATH) as f:
            return json.load(f)

    current_user_id = _call_with_retry(sp.current_user).get("id")

    print("Fetching playlists from Spotify API...")
    playlists = []
    offset = 0
    while True:
        results = _call_with_retry(
            sp.current_user_playlists, limit=PLAYLISTS_PAGE_SIZE, offset=offset
        )
        batch = results.get("items", [])
        if not batch:
            break
        playlists.extend(batch)
        if len(batch) < PLAYLISTS_PAGE_SIZE:
            break
        offset += PLAYLISTS_PAGE_SIZE

    output = []
    for pl in playlists:
        print(f"  fetching tracks for playlist '{pl.get('name')}'...")
        try:
            track_ids = _fetch_playlist_track_ids(sp, pl["id"])
        except SpotifyException as e:
            if e.http_status == 403:
                print(f"    Skipping playlist '{pl.get('name')}' — not owned/collaborative, contents unavailable")
                track_ids = []
            else:
                raise
        images = pl.get("images") or []
        output.append(
            {
                "id": pl["id"],
                "name": pl.get("name"),
                "image_url": images[0]["url"] if images else None,
                "is_owned": (pl.get("owner") or {}).get("id") == current_user_id,
                "track_ids": track_ids,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_PLAYLISTS_PATH, "w") as f:
        json.dump(output, f)
    return output


def fetch_saved_albums(sp: spotipy.Spotify, force_refresh: bool = False) -> list:
    """Fetch all of the user's saved albums, paginating 50 at a time. Cached to disk.

    Returns a list of {"id", "name", "artist_name", "image_url", "track_count",
    "track_ids"} dicts. Album track IDs come from the nested tracks.items in the
    saved-album response; very large albums beyond that single page won't have
    every track ID captured, but that's a rare edge case for personal libraries.
    """
    if not force_refresh and RAW_ALBUMS_PATH.exists():
        print(f"Loading saved albums from cache ({RAW_ALBUMS_PATH})...")
        with open(RAW_ALBUMS_PATH) as f:
            return json.load(f)

    print("Fetching saved albums from Spotify API...")
    items = []
    offset = 0
    while True:
        results = _call_with_retry(
            sp.current_user_saved_albums, limit=SAVED_ALBUMS_PAGE_SIZE, offset=offset
        )
        batch = results.get("items", [])
        if not batch:
            break
        items.extend(batch)
        print(f"  fetched {len(items)} albums so far...")
        if len(batch) < SAVED_ALBUMS_PAGE_SIZE:
            break
        offset += SAVED_ALBUMS_PAGE_SIZE

    output = []
    for entry in items:
        album = entry.get("album") or {}
        track_items = ((album.get("tracks") or {}).get("items")) or []
        track_ids = [t["id"] for t in track_items if t and t.get("id")]
        images = album.get("images") or []
        output.append(
            {
                "id": album.get("id"),
                "name": album.get("name"),
                "artist_name": ", ".join(a["name"] for a in album.get("artists", [])),
                "image_url": images[0]["url"] if images else None,
                "track_count": album.get("total_tracks") or len(track_ids),
                "track_ids": track_ids,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_ALBUMS_PATH, "w") as f:
        json.dump(output, f)
    return output


def remove_track_from_playlist(sp: spotipy.Spotify, playlist_id: str, track_id: str) -> bool:
    """Remove all occurrences of a track from a playlist (requires playlist-modify-*).

    Never raises — returns True on success, False (with a printed warning) on
    failure, same pattern as unsave_track.
    """
    try:
        _call_with_retry(sp.playlist_remove_all_occurrences_of_items, playlist_id, [track_id])
        return True
    except SpotifyException as e:
        if e.http_status == 403:
            print(
                f"WARNING: failed to remove track {track_id} from playlist {playlist_id} "
                "(403 Forbidden). This usually means the app's OAuth token doesn't have "
                "the playlist-modify-public/private scope yet, or you don't own this "
                "playlist — delete .spotify_token_cache and restart to re-authorize."
            )
        else:
            print(f"WARNING: failed to remove track {track_id} from playlist {playlist_id}: {e}")
        return False


def unsave_track(sp: spotipy.Spotify, track_id: str) -> bool:
    """Remove a track from the user's Liked Songs library (requires user-library-modify).

    Never raises — returns True on success, False (with a printed warning) on
    failure, so a failed unsave doesn't crash the caller or block a label from
    being recorded.
    """
    try:
        _call_with_retry(sp.current_user_saved_tracks_delete, [track_id])
        return True
    except SpotifyException as e:
        if e.http_status == 403:
            print(
                f"WARNING: failed to unsave track {track_id} (403 Forbidden). "
                "This usually means the app's OAuth token doesn't have the "
                "user-library-modify scope yet — delete .spotify_token_cache and "
                "restart to re-authorize with the new permission."
            )
        else:
            print(f"WARNING: failed to unsave track {track_id} from your library: {e}")
        return False
