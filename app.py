"""Phase 2: Streamlit swipe UI with an active-learning (uncertainty sampling) loop.

Two-screen flow: a gallery of the user's owned playlists/saved albums/all Liked
Songs, then a swipe session scoped to whichever one they pick. Labels always
accumulate into one global data/labels.csv regardless of which collection they
came from, and the active-learning model always trains on all of them — only
the *candidate pool* shown next is scoped to the current collection.

The first BOOTSTRAP_SIZE labels (globally, across all collections) are random
order; after that, each swipe retrains a RandomForestClassifier on all labels
so far and picks whichever unlabeled track (within the current collection) the
model is most uncertain about. Every swipe's (num_labels, k-fold accuracy) is
logged to data/learning_curve.csv.

Remove is a live action against Spotify, and what it does depends on context:
- "All Liked Songs": unsaves the track from Liked Songs (as before).
- A specific owned playlist: removes the track from that playlist by default;
  a sidebar toggle can make it also unsave from Liked Songs.
- A saved album: albums can't have individual tracks removed as saved units,
  so Remove falls back to unsaving the track from Liked Songs instead.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from spotify_client import (
    DATA_DIR,
    fetch_playlists_with_tracks,
    fetch_saved_albums,
    get_spotify_client,
    remove_track_from_playlist,
    unsave_track,
)

TRACKS_CSV_PATH = DATA_DIR / "tracks.csv"
LABELS_CSV_PATH = DATA_DIR / "labels.csv"
LEARNING_CURVE_CSV_PATH = DATA_DIR / "learning_curve.csv"

BOOTSTRAP_SIZE = 20
GALLERY_COLUMNS_PER_ROW = 4

NUMERIC_SOURCE_COLUMNS = [
    "release_year",
    "popularity",
    "days_since_added",
    "play_recency_days",
    "playlist_count",
    "artist_track_count",
]
FEATURE_COLUMNS = NUMERIC_SOURCE_COLUMNS


# --- Data loading / feature prep (unchanged from before this change) -------

def load_tracks() -> pd.DataFrame:
    return pd.read_csv(TRACKS_CSV_PATH)


def load_labels(tracks_columns: list) -> pd.DataFrame:
    if LABELS_CSV_PATH.exists():
        return pd.read_csv(LABELS_CSV_PATH)
    return pd.DataFrame(columns=tracks_columns + ["label"])


def compute_fill_values(tracks_df: pd.DataFrame) -> dict:
    fills = {}
    for col in NUMERIC_SOURCE_COLUMNS:
        numeric = pd.to_numeric(tracks_df[col], errors="coerce")
        fills[col] = numeric.median() if numeric.notna().any() else 0
    max_recency = pd.to_numeric(tracks_df["play_recency_days"], errors="coerce").max()
    fills["play_recency_days"] = (max_recency + 1) if pd.notna(max_recency) else 9999.0
    return fills


def prepare_features(df: pd.DataFrame, fills: dict) -> pd.DataFrame:
    """Numeric feature matrix indexed by track_id, using a shared fill mapping so
    imputed values mean the same thing whether called on labeled data (for
    training) or unlabeled data (for scoring)."""
    feats = df.copy()
    for col in NUMERIC_SOURCE_COLUMNS:
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(fills[col])
    return feats.set_index("track_id")[FEATURE_COLUMNS]


def k_fold_accuracy(X: np.ndarray, y: np.ndarray, max_splits: int = 5):
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_splits = min(max_splits, int(class_counts.min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=skf)
    return float(scores.mean())


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def pick_next_track(tracks_df, labeled_ids, fills, model=None):
    remaining = tracks_df[~tracks_df["track_id"].isin(labeled_ids)]
    if remaining.empty:
        return None
    if model is None:
        return remaining["track_id"].sample(1).iloc[0]

    feats = prepare_features(remaining, fills)
    class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    probs = model.predict_proba(feats.values)[:, class_idx]
    uncertainty = np.abs(probs - 0.5)
    return feats.index[int(np.argmin(uncertainty))]


def append_csv_row(path, row_df: pd.DataFrame):
    write_header = not path.exists()
    row_df.to_csv(path, mode="a", header=write_header, index=False)


# --- Gallery (Screen 1) -----------------------------------------------------

def build_gallery_items(sp) -> list:
    """Owned playlists + saved albums, normalized to a common shape for the gallery."""
    items = []

    for pl in fetch_playlists_with_tracks(sp):
        if not pl.get("is_owned"):
            continue
        items.append(
            {
                "type": "playlist",
                "id": pl["id"],
                "name": pl["name"],
                "image_url": pl.get("image_url"),
                "track_count": len(pl["track_ids"]),
                "track_ids": pl["track_ids"],
                "is_owned": True,
            }
        )

    for al in fetch_saved_albums(sp):
        items.append(
            {
                "type": "album",
                "id": al["id"],
                "name": f"{al['name']} — {al.get('artist_name', '')}",
                "image_url": al.get("image_url"),
                "track_count": al.get("track_count", len(al["track_ids"])),
                "track_ids": al["track_ids"],
                "is_owned": True,
            }
        )

    return items


def render_gallery():
    st.title("What do you want to swipe through?")

    if st.session_state.gallery_items is None:
        with st.spinner("Loading your playlists and albums..."):
            st.session_state.gallery_items = build_gallery_items(st.session_state.sp)

    tracks_df = st.session_state.tracks_df
    liked_item = {
        "type": "liked",
        "id": None,
        "name": "All Liked Songs",
        "image_url": None,
        "track_count": len(tracks_df),
        "track_ids": tracks_df["track_id"].tolist(),
        "is_owned": True,
    }
    all_items = [liked_item] + st.session_state.gallery_items

    for row_start in range(0, len(all_items), GALLERY_COLUMNS_PER_ROW):
        row_items = all_items[row_start : row_start + GALLERY_COLUMNS_PER_ROW]
        cols = st.columns(GALLERY_COLUMNS_PER_ROW)
        for col, item in zip(cols, row_items):
            with col:
                if item["image_url"]:
                    st.image(item["image_url"], use_container_width=True)
                else:
                    st.markdown("🎵")
                if st.button(f"{item['name']}\n({item['track_count']} tracks)", key=f"select_{item['type']}_{item['id']}"):
                    st.session_state.selected_collection = item
                    st.session_state.current_track_id = None
                    st.rerun()


# --- Swipe session (Screen 2) ----------------------------------------------

def handle_remove_action(sp, collection: dict, track_id: str):
    """Dispatch Remove based on the selected collection's type."""
    if collection["type"] == "liked":
        if unsave_track(sp, track_id):
            st.toast("Removed from your library", icon="✅")
        else:
            st.warning(
                "Couldn't unsave this track from your Spotify library (label was "
                "still recorded). If this keeps happening, delete "
                ".spotify_token_cache and restart the app to re-authorize."
            )
        return

    if collection["type"] == "album":
        if unsave_track(sp, track_id):
            st.toast("Removed from your saved tracks", icon="✅")
        else:
            st.warning("Couldn't unsave this track (label was still recorded).")
        st.caption("Removed from your saved tracks — a single track can't be removed from a saved album.")
        return

    # playlist
    if not collection.get("is_owned"):
        st.warning("Remove is disabled for this playlist — you don't own it, so its contents can't be modified.")
        return

    mode = st.session_state.get("remove_mode", "playlist")

    if remove_track_from_playlist(sp, collection["id"], track_id):
        st.toast(f"Removed from '{collection['name']}'", icon="✅")
    else:
        st.warning(f"Couldn't remove this track from '{collection['name']}' (label was still recorded).")

    if mode == "library":
        if unsave_track(sp, track_id):
            st.toast("Also unsaved from your library", icon="✅")
        else:
            st.warning("Couldn't also unsave this track from your library (label was still recorded).")


def render_swipe_session():
    collection = st.session_state.selected_collection
    tracks_df = st.session_state.tracks_df
    labels_df = st.session_state.labels_df
    fills = st.session_state.fills
    sp = st.session_state.sp

    collection_tracks_df = tracks_df[tracks_df["track_id"].isin(collection["track_ids"])].reset_index(drop=True)
    labeled_ids = set(labels_df["track_id"]) if not labels_df.empty else set()

    with st.sidebar:
        st.header("Progress")
        st.metric("Labels so far", len(labels_df))
        if len(labels_df) < BOOTSTRAP_SIZE:
            st.caption(f"Bootstrap phase: {len(labels_df)}/{BOOTSTRAP_SIZE} random swipes.")
        if st.session_state.last_accuracy is not None:
            st.metric("Model accuracy (k-fold)", f"{st.session_state.last_accuracy:.1%}")
        else:
            st.caption("Accuracy appears once k-fold CV is possible (both labels present).")

        st.divider()
        if st.button("← Back to gallery"):
            st.session_state.selected_collection = None
            st.session_state.current_track_id = None
            st.rerun()

        if collection["type"] == "playlist" and collection.get("is_owned"):
            st.radio(
                "Remove means:",
                options=["playlist", "library"],
                format_func=lambda v: "Remove from this playlist" if v == "playlist" else "Also unsave from Liked Songs",
                key="remove_mode",
            )

    st.title(f"Keep or Remove? — {collection['name']}")

    if collection_tracks_df.empty:
        st.warning(
            "None of this collection's tracks are in your saved library yet, so "
            "there's nothing to swipe here — the swipe queue only draws from your "
            "already-saved tracks (data/tracks.csv)."
        )
        return

    if st.session_state.current_track_id is None or st.session_state.current_track_id in labeled_ids:
        model = None
        if len(labeled_ids) >= BOOTSTRAP_SIZE and len(labels_df) > 0:
            feats = prepare_features(labels_df, fills)
            model = train_model(feats.values, labels_df["label"].astype(int).values)
        st.session_state.current_track_id = pick_next_track(
            collection_tracks_df, labeled_ids, fills, model=model
        )

    current_id = st.session_state.current_track_id
    if current_id is None:
        st.success(f"All {len(collection_tracks_df)} tracks in '{collection['name']}' labeled!")
        return

    track = collection_tracks_df[collection_tracks_df["track_id"] == current_id].iloc[0]

    if pd.notna(track.get("image_url")):
        st.image(track["image_url"], width=300)
    st.subheader(track["track_name"])
    st.caption(f"{track['artist_name']} — {track.get('album_name', '')}")

    col1, col2 = st.columns(2)
    keep_clicked = col1.button("Keep", key="keep_btn")
    remove_clicked = col2.button("Remove", key="remove_btn")

    if keep_clicked or remove_clicked:
        label = 1 if keep_clicked else 0
        row = track.to_dict()
        row["label"] = label
        row_df = pd.DataFrame([row])

        st.session_state.labels_df = pd.concat(
            [st.session_state.labels_df, row_df], ignore_index=True
        )
        append_csv_row(LABELS_CSV_PATH, row_df)

        if remove_clicked:
            handle_remove_action(sp, collection, current_id)

        updated_labels_df = st.session_state.labels_df
        feats = prepare_features(updated_labels_df, fills)
        y = updated_labels_df["label"].astype(int).values
        accuracy = k_fold_accuracy(feats.values, y)
        st.session_state.last_accuracy = accuracy

        log_row = pd.DataFrame([{"num_labels": len(updated_labels_df), "accuracy": accuracy}])
        append_csv_row(LEARNING_CURVE_CSV_PATH, log_row)

        st.session_state.current_track_id = None
        st.rerun()


def main():
    st.set_page_config(page_title="Spotify Swipe", layout="centered")

    if not TRACKS_CSV_PATH.exists():
        st.error("data/tracks.csv not found. Run `python features.py` first.")
        st.stop()

    if "tracks_df" not in st.session_state:
        st.session_state.tracks_df = load_tracks()
        st.session_state.labels_df = load_labels(list(st.session_state.tracks_df.columns))
        st.session_state.fills = compute_fill_values(st.session_state.tracks_df)
        st.session_state.current_track_id = None
        st.session_state.last_accuracy = None
        st.session_state.sp = get_spotify_client()
        st.session_state.selected_collection = None
        st.session_state.gallery_items = None

    if st.session_state.selected_collection is None:
        render_gallery()
    else:
        render_swipe_session()


if __name__ == "__main__":
    main()
