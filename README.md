# Spotify ML Portfolio Project

An ML portfolio project built on top of your own Spotify listening data.

- **Phase 1** (`spotify_client.py`, `features.py`): pulls your saved tracks plus
  metadata and behavioral signals (recency, playlist co-occurrence, artist
  frequency) into one clean, cached `data/tracks.csv`.
- **Phase 2** (`app.py`): a Streamlit swipe UI (Keep/Remove) that labels tracks into
  `data/labels.csv` and, after a short bootstrap, uses active learning (uncertainty
  sampling with a RandomForestClassifier) to pick which track to show next. Remove
  is a live action — it also unsaves the track from your actual Spotify library.

Not part of this repo yet: Phase 3 (model evaluation/learning-curve charts) and
Phase 4 (a write-up of what the model learned).

Audio-features (danceability, energy, valence, etc.), `preview_url`, and artist
genre data are **not** used anywhere in this project — all three require Spotify
endpoints (`audio-features`, `artists`) that 403 for apps without Extended Quota
Mode approval, which personal/hobby apps don't get. All signal instead comes from
track metadata and your own behavior (recently played, playlists, saves).

## Setup

### 1. Get Spotify API credentials

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   and log in with your Spotify account.
2. Click **Create app**.
3. Fill in a name/description (anything you like).
4. Under **Redirect URIs**, add exactly: `http://127.0.0.1:8888/callback`
   (Spotify no longer accepts `localhost` as of Feb 2026 — it must be a loopback
   IP literal.)
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
| `SPOTIFY_REDIRECT_URI` | Must match the dashboard exactly: `http://127.0.0.1:8888/callback` |

`.env` is gitignored — never commit real credentials.

### 3. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Phase 1 — the data pipeline

```bash
python features.py
```

On first run, a browser window will open asking you to log in to Spotify and
authorize the app (scopes: `user-library-read`, `user-library-modify`,
`user-read-recently-played`, `user-top-read`, `playlist-read-private`,
`playlist-read-collaborative`). After authorizing, you'll be redirected to
`http://127.0.0.1:8888/callback?code=...` — that page won't load (nothing is
listening on that port yet), which is expected; spotipy reads the `code` from
the URL and continues. Your auth token is then cached locally in
`.spotify_token_cache` so you won't have to log in again on future runs.

The script will:
1. Fetch all your saved tracks (paginated, cached to `data/raw_tracks.json`).
2. Fetch your recently played tracks (cached to `data/raw_recently_played.json`).
3. Fetch your playlists and each playlist's track IDs (cached to
   `data/raw_playlists.json`). Playlists you don't own or collaborate on will be
   skipped with a printed note — Spotify no longer returns their contents.
4. Combine everything into one dataframe and save it to `data/tracks.csv`.
5. Print a summary: total tracks pulled, how many made it into the final
   dataframe, and how many were dropped (local files/podcasts have no usable
   track ID).

Re-running `python features.py` reuses the cached JSON files instead of
re-hitting the Spotify API. Delete the relevant file(s) in `data/`, or pass
`force_refresh=True` to `run_pipeline()`, to force a refetch.

## Running Phase 2 — the swipe UI

```bash
streamlit run app.py
```

Requires `data/tracks.csv` to already exist (run Phase 1 first). Swipe Keep/Remove
on tracks one at a time; the first 20 swipes are in random order to bootstrap a
model, after which each swipe retrains a RandomForestClassifier on your labels so
far and picks the next track it's most uncertain about. Labels are saved to
`data/labels.csv`, and the accuracy of the model at each step is logged to
`data/learning_curve.csv`.

**Remove is a live action against your Spotify account.** Clicking Remove both
logs the label locally *and* calls the Spotify API to unsave that track from your
Liked Songs library immediately — it is not a dry run. If the app's cached OAuth
token predates the `user-library-modify` scope, delete `.spotify_token_cache` and
restart to re-authorize; a failed unsave shows a warning in the UI but the label
is still recorded either way.
