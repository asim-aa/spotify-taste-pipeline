"""Phase 2: Streamlit swipe UI with an active-learning (uncertainty sampling) loop.

Labels 1 track at a time as Keep/Remove. The first BOOTSTRAP_SIZE swipes are in
random order to give the classifier something to start from; after that, each
swipe retrains a RandomForestClassifier on data/labels.csv and the next track
shown is whichever unlabeled track the model is most uncertain about (predicted
P(keep) closest to 0.5). Every swipe's (num_labels, k-fold accuracy) is logged to
data/learning_curve.csv for a later active-learning-vs-random comparison chart.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from spotify_client import DATA_DIR

TRACKS_CSV_PATH = DATA_DIR / "tracks.csv"
LABELS_CSV_PATH = DATA_DIR / "labels.csv"
LEARNING_CURVE_CSV_PATH = DATA_DIR / "learning_curve.csv"

BOOTSTRAP_SIZE = 20

NUMERIC_SOURCE_COLUMNS = [
    "release_year",
    "popularity",
    "days_since_added",
    "play_recency_days",
    "playlist_count",
    "artist_track_count",
]
FEATURE_COLUMNS = NUMERIC_SOURCE_COLUMNS


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

    tracks_df = st.session_state.tracks_df
    labels_df = st.session_state.labels_df
    fills = st.session_state.fills
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

    if st.session_state.current_track_id is None or st.session_state.current_track_id in labeled_ids:
        model = None
        if len(labeled_ids) >= BOOTSTRAP_SIZE and len(labels_df) > 0:
            feats = prepare_features(labels_df, fills)
            model = train_model(feats.values, labels_df["label"].astype(int).values)
        st.session_state.current_track_id = pick_next_track(
            tracks_df, labeled_ids, fills, model=model
        )

    current_id = st.session_state.current_track_id
    if current_id is None:
        st.success(f"All {len(tracks_df)} tracks labeled!")
        st.stop()

    track = tracks_df[tracks_df["track_id"] == current_id].iloc[0]

    st.title("Keep or Remove?")
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

        updated_labels_df = st.session_state.labels_df
        feats = prepare_features(updated_labels_df, fills)
        y = updated_labels_df["label"].astype(int).values
        accuracy = k_fold_accuracy(feats.values, y)
        st.session_state.last_accuracy = accuracy

        log_row = pd.DataFrame([{"num_labels": len(updated_labels_df), "accuracy": accuracy}])
        append_csv_row(LEARNING_CURVE_CSV_PATH, log_row)

        st.session_state.current_track_id = None
        st.rerun()


if __name__ == "__main__":
    main()
