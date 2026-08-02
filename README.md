# Spotify ML Portfolio Project

An ML portfolio project built on top of your own Spotify listening data. This is
Phase 1: a data pipeline that pulls your saved tracks and audio features into one
clean, cached `data/tracks.csv`. Later phases add a Streamlit active-learning swipe
UI (`app.py`) and a classifier (`model.py`) — not part of this phase.

## Setup

### 1. Get Spotify API credentials

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and log in with your Spotify account.
2. Click **Create app**.
3. Fill in a name/description (anything you like).
4. Under **Redirect URIs**, add exactly: `http://localhost:8501`
5. Check the box for the Web API, agree to the terms, and click **Save**.
6. Open the app you just created, click **Settings**, and copy the **Client ID**
   and **Client Secret**.

### 2. Configure environment variables

Copy the example env file and fill in the values from the previous step:

```bash
cp .env.example .env
```

`.env` needs:

| Variable | Description |
|---|---|
| `SPOTIFY_CLIENT_ID` | Client ID from the Spotify Developer Dashboard |
| `SPOTIFY_CLIENT_SECRET` | Client Secret from the Spotify Developer Dashboard |
| `SPOTIFY_REDIRECT_URI` | Must match the dashboard exactly: `http://localhost:8501` |

`.env` is gitignored — never commit real credentials.

### 3. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
python features.py
```

On first run, a browser window will open asking you to log in to Spotify and
authorize the app (scopes: `user-library-read`, `user-read-recently-played`,
`user-top-read`). After authorizing, you'll be redirected to
`http://localhost:8501/?code=...` — that page won't load (nothing is listening
on that port yet), which is expected; spotipy reads the `code` from the URL
and continues. Your auth token is then cached locally in
`.spotify_token_cache` so you won't have to log in again on future runs.

The script will:
1. Fetch all your saved tracks (paginated, cached to `data/raw_tracks.json`).
2. Fetch audio features for those tracks (batched, cached to
   `data/raw_audio_features.json`).
3. Combine everything into one dataframe and save it to `data/tracks.csv`.
4. Print a summary: total tracks pulled, how many had complete audio features,
   and how many were dropped (local files/podcasts and tracks missing audio
   features are dropped rather than crashing the pipeline).

Re-running `python features.py` reuses the cached JSON files instead of
re-hitting the Spotify API. Delete the relevant file(s) in `data/`, or pass
`force_refresh=True` to `run_pipeline()`, to force a refetch.
