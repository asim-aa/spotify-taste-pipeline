"""Phase 3: model evaluation, active-learning-vs-random comparison, and feature importance.

Reads only from local data/ CSVs (tracks.csv, labels.csv, learning_curve.csv) — no
Spotify API calls. Reuses app.py's feature prep and k-fold logic so these numbers
are directly comparable to what the swipe UI reported live.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app import (
    FEATURE_COLUMNS,
    compute_fill_values,
    k_fold_accuracy,
    load_tracks,
    prepare_features,
    train_model,
)
from spotify_client import DATA_DIR

LABELS_CSV_PATH = DATA_DIR / "labels.csv"
LEARNING_CURVE_CSV_PATH = DATA_DIR / "learning_curve.csv"
COMPARISON_CHART_PATH = DATA_DIR / "labels_vs_accuracy.png"
FEATURE_IMPORTANCE_CHART_PATH = DATA_DIR / "feature_importance.png"
MODEL_COMPARISON_CHART_PATH = DATA_DIR / "model_comparison.png"
TASTE_SUMMARY_PATH = DATA_DIR / "taste_summary.md"
REMOVAL_ACTION_CHART_PATH = DATA_DIR / "removal_action_breakdown.png"
KEEP_RATE_BY_COLLECTION_CHART_PATH = DATA_DIR / "keep_rate_by_collection.png"
SWIPE_ACTIVITY_CHART_PATH = DATA_DIR / "swipe_activity_over_time.png"

RANDOM_TRIALS = 20
MIN_LABELS_FOR_CONFIDENCE = 100

# Features whose raw distributions are prone to outliers/skew — compare medians
# instead of means for these so a couple of extreme tracks don't dominate.
OUTLIER_PRONE_FEATURES = {"days_since_added", "release_year"}

FEATURE_DESCRIPTIONS = {
    "days_since_added": "how recently you added the track",
    "artist_track_count": "how many other saved tracks you have from the same artist",
    "release_year": "how old the track is",
    "playlist_count": "how many of your playlists include the track",
    "play_recency_days": "how recently you've actually played it",
    "popularity": "Spotify's popularity score",
}

# Natural-language phrase for "kept tracks skew lower/higher on this feature."
FEATURE_DIRECTION_PHRASES = {
    "days_since_added": {
        "lower": "tracks you added more recently",
        "higher": "tracks you added a while ago",
    },
    "artist_track_count": {
        "lower": "tracks by artists you've saved comparatively few tracks from",
        "higher": "tracks by artists you've saved a lot of tracks from",
    },
    "release_year": {
        "lower": "older tracks",
        "higher": "newer tracks",
    },
    "playlist_count": {
        "lower": "tracks that aren't in many of your playlists",
        "higher": "tracks that show up in more of your playlists",
    },
    "play_recency_days": {
        "lower": "tracks you've played more recently",
        "higher": "tracks you haven't played in a while",
    },
}

FEATURE_VALUE_FORMAT = {
    "days_since_added": lambda v: f"{v:.0f} days ago",
    "artist_track_count": lambda v: f"{v:.1f} tracks",
    "release_year": lambda v: f"{v:.0f}",
    "playlist_count": lambda v: f"{v:.1f} playlists",
    "play_recency_days": lambda v: f"{v:.0f} days ago",
}


def load_labels_and_fills() -> tuple[pd.DataFrame, dict]:
    tracks_df = load_tracks()
    fills = compute_fill_values(tracks_df)
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    return labels_df, fills


def majority_baseline_accuracy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def evaluate_final_model(labels_df: pd.DataFrame, fills: dict):
    feats = prepare_features(labels_df, fills)
    X = feats.values
    y = labels_df["label"].astype(int).values

    model = train_model(X, y)
    accuracy = k_fold_accuracy(X, y)
    baseline = majority_baseline_accuracy(y)

    print("=" * 60)
    print("FINAL MODEL vs BASELINE")
    print("=" * 60)
    if accuracy is None:
        print("WARNING: not enough examples per class for k-fold CV (need at least")
        print("2 examples of each label per fold) — skipping model/baseline comparison.")
    else:
        lift_pts = (accuracy - baseline) * 100
        sign = "+" if lift_pts >= 0 else ""
        print(f"  Model (k-fold CV):       {accuracy:.1%}")
        print(f"  Majority-class baseline: {baseline:.1%}")
        print(f"  Lift: {sign}{lift_pts:.1f}pts")
    print()

    return model, accuracy, baseline


# --- Logistic regression comparison (additive — RandomForest above remains the
# model used everywhere else: the live active-learning loop in app.py, the
# feature-importance chart, and taste-summary generation. RF was chosen there
# for its non-linear splits and the feature_importances_-based interpretability
# story; logistic regression is included here purely as a linear-model
# sanity-check baseline, not a replacement. ---------------------------------

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Same signature/pattern as train_model() above, for a RandomForest.
    max_iter raised from sklearn's default of 100, which often fails to
    converge on a feature set this small; random_state matches the RF's for
    a like-for-like comparison."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def k_fold_accuracy_for_model(model_factory, X: np.ndarray, y: np.ndarray, max_splits: int = 5):
    """Model-agnostic twin of app.py's k_fold_accuracy, which is hardcoded to
    RandomForestClassifier internally (intentionally left untouched there,
    since it's used live by the active-learning loop). model_factory is a
    zero-arg callable returning a fresh unfitted estimator.

    Uses the identical StratifiedKFold(shuffle=True, random_state=42)
    construction as app.py's k_fold_accuracy — given the same X/y, that
    produces the exact same fold assignments, so calling this with
    LogisticRegression and calling app.py's k_fold_accuracy with the same X/y
    are scored on identical splits, not just compared as independent numbers.
    """
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_splits = min(max_splits, int(class_counts.min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model_factory(), X, y, cv=skf)
    return float(scores.mean())


def evaluate_logistic_regression(labels_df: pd.DataFrame, fills: dict):
    """Train + k-fold-score a LogisticRegression on the exact same feature
    prep as evaluate_final_model, so the two models are compared on identical
    data. Returns (model, accuracy) — does not print; see print_model_comparison."""
    feats = prepare_features(labels_df, fills)
    X = feats.values
    y = labels_df["label"].astype(int).values

    model = train_logistic_regression(X, y)
    accuracy = k_fold_accuracy_for_model(
        lambda: LogisticRegression(max_iter=1000, random_state=42), X, y
    )
    return model, accuracy


def print_model_comparison(rf_accuracy, logreg_accuracy, baseline: float):
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"  Majority-class baseline:  {baseline:.1%}")

    if rf_accuracy is None or logreg_accuracy is None:
        print("  WARNING: not enough examples per class for k-fold CV — skipping model comparison.")
        print("=" * 60)
        print()
        return

    logreg_lift = (logreg_accuracy - baseline) * 100
    rf_lift = (rf_accuracy - baseline) * 100
    logreg_sign = "+" if logreg_lift >= 0 else ""
    rf_sign = "+" if rf_lift >= 0 else ""

    print(f"  Logistic Regression:      {logreg_accuracy:.1%}   ({logreg_sign}{logreg_lift:.1f}pts vs baseline)")
    print(f"  Random Forest:            {rf_accuracy:.1%}   ({rf_sign}{rf_lift:.1f}pts vs baseline)")
    print()

    diff_pts = (rf_accuracy - logreg_accuracy) * 100
    if abs(diff_pts) < 0.05:
        print("  Winner: tie")
    elif diff_pts > 0:
        print(f"  Winner: RandomForest by {diff_pts:.1f}pts")
    else:
        print(f"  Winner: LogisticRegression by {abs(diff_pts):.1f}pts")
    print("=" * 60)
    print()


def print_logistic_regression_coefficients(model: LogisticRegression, feature_names: list):
    """Unlike RF's feature_importances_ (magnitude-only, always positive),
    logistic regression coefficients are directly interpretable: sign shows
    direction. Positive pushes toward Keep (label=1), negative toward Remove.
    Note: features aren't standardized before fitting (matching train_model()'s
    simplicity), so raw coefficient MAGNITUDE isn't directly comparable across
    features on very different scales (e.g. release_year vs playlist_count) —
    direction is still valid regardless."""
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]

    print("=" * 60)
    print("LOGISTIC REGRESSION COEFFICIENTS (not RandomForest)")
    print("=" * 60)
    for i in order:
        if coefs[i] > 0:
            direction = "-> pushes toward Keep"
        elif coefs[i] < 0:
            direction = "-> pushes toward Remove"
        else:
            direction = "-> no effect"
        print(f"  {feature_names[i]:<20s} {coefs[i]:+.4f}  {direction}")
    print("  (magnitude not directly comparable across features — different scales; sign is what matters)")
    print()


def plot_model_comparison(baseline: float, logreg_accuracy, rf_accuracy):
    """Optional bar chart: baseline vs logreg vs RF accuracy. Skips gracefully
    if either model's k-fold CV wasn't possible."""
    if logreg_accuracy is None or rf_accuracy is None:
        print("Skipping model comparison chart — not enough data for k-fold CV on one or both models.")
        return

    labels = ["Baseline", "Logistic\nRegression", "Random\nForest"]
    values = [baseline * 100, logreg_accuracy * 100, rf_accuracy * 100]
    colors = ["#777777", "#F5A623", "#1DB954"]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Model Comparison: Baseline vs. Logistic Regression vs. Random Forest")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_CHART_PATH, dpi=150)
    plt.close()
    print(f"Saved model comparison chart to {MODEL_COMPARISON_CHART_PATH}")


def simulate_random_curve(labels_df: pd.DataFrame, fills: dict, n_trials: int = RANDOM_TRIALS):
    """Average k-fold accuracy at each label count under random shuffles of the
    already-collected labels, as a stand-in for what pure random sampling would
    have looked like over the same labeling effort."""
    feats = prepare_features(labels_df, fills)
    X_full = feats.values
    y_full = labels_df["label"].astype(int).values
    n = len(labels_df)

    rng = np.random.default_rng(42)
    accuracy_sums: dict = {}
    accuracy_counts: dict = {}

    for _ in range(n_trials):
        perm = rng.permutation(n)
        X_shuffled = X_full[perm]
        y_shuffled = y_full[perm]
        for i in range(1, n + 1):
            acc = k_fold_accuracy(X_shuffled[:i], y_shuffled[:i])
            if acc is None:
                continue
            accuracy_sums[i] = accuracy_sums.get(i, 0.0) + acc
            accuracy_counts[i] = accuracy_counts.get(i, 0) + 1

    steps = sorted(accuracy_sums)
    avg_accuracy = [accuracy_sums[i] / accuracy_counts[i] for i in steps]
    return steps, avg_accuracy


def load_real_learning_curve():
    df = pd.read_csv(LEARNING_CURVE_CSV_PATH).dropna(subset=["accuracy"])
    return df["num_labels"].tolist(), df["accuracy"].tolist()


def plot_comparison(real_steps, real_acc, random_steps, random_acc):
    plt.figure(figsize=(9, 6))
    plt.plot(real_steps, real_acc, marker="o", label="Active learning (real session)", color="#1DB954")
    plt.plot(
        random_steps,
        random_acc,
        marker="o",
        label=f"Random sampling (avg of {RANDOM_TRIALS} shuffles)",
        color="#777777",
    )
    plt.xlabel("Number of labels")
    plt.ylabel("Model accuracy (k-fold CV)")
    plt.title("Active Learning vs. Random Sampling")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARISON_CHART_PATH, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_importances = [importances[i] for i in order]

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    for name, imp in zip(sorted_names, sorted_importances):
        print(f"  {name:<20s} {imp:.4f}")
    print()

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_names[::-1], sorted_importances[::-1], color="#1DB954")
    plt.xlabel("Importance")
    plt.title("Feature Importance (Final RandomForest Model)")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_CHART_PATH, dpi=150)
    plt.close()

    return sorted_names, sorted_importances


def _feature_group_stat(values: pd.Series, feature: str) -> tuple[float, str]:
    """Median for outlier-prone features, mean otherwise. Returns (value, method_name)."""
    if feature in OUTLIER_PRONE_FEATURES:
        return float(values.median()), "median"
    return float(values.mean()), "average"


def generate_taste_summary(labels_df: pd.DataFrame, sorted_names: list, sorted_importances: list):
    """Turn the top (nonzero-importance) features into plain-English sentences about
    which direction of each feature correlates with Keep vs. Remove, using the
    labels actually collected. Prints the section and saves it to TASTE_SUMMARY_PATH."""
    keep_df = labels_df[labels_df["label"] == 1]
    remove_df = labels_df[labels_df["label"] == 0]

    top_features = [
        (name, imp)
        for name, imp in zip(sorted_names, sorted_importances)
        if imp > 0 and name in FEATURE_DIRECTION_PHRASES
    ][:3]

    sentences = []
    for feature, _ in top_features:
        keep_vals = pd.to_numeric(keep_df[feature], errors="coerce").dropna()
        remove_vals = pd.to_numeric(remove_df[feature], errors="coerce").dropna()
        if keep_vals.empty or remove_vals.empty:
            continue

        keep_stat, method = _feature_group_stat(keep_vals, feature)
        remove_stat, _ = _feature_group_stat(remove_vals, feature)
        fmt = FEATURE_VALUE_FORMAT.get(feature, lambda v: f"{v:.1f}")
        article = "an" if method[0] in "aeiou" else "a"

        if np.isclose(keep_stat, remove_stat):
            sentence = (
                f"{FEATURE_DESCRIPTIONS[feature].capitalize()} showed no clear difference "
                f"between kept and removed tracks ({fmt(keep_stat)} vs {fmt(remove_stat)})."
            )
        else:
            direction = "lower" if keep_stat < remove_stat else "higher"
            phrase = FEATURE_DIRECTION_PHRASES[feature][direction]
            sentence = (
                f"You tend to keep {phrase}, based on {article} {method} of {fmt(keep_stat)} for "
                f"kept tracks vs {fmt(remove_stat)} for removed ones."
            )
        sentences.append(sentence)

    print()
    print("=" * 60)
    print("WHAT THE MODEL LEARNED ABOUT YOUR TASTE")
    print("=" * 60)

    md_lines = ["# What the Model Learned About Your Taste", ""]

    if not sentences:
        msg = "Not enough signal yet to describe a taste pattern — keep swiping."
        print(msg)
        md_lines.append(msg)
    else:
        for sentence in sentences:
            print(f"- {sentence}")
            md_lines.append(f"- {sentence}")

    if len(labels_df) < MIN_LABELS_FOR_CONFIDENCE:
        caveat = (
            f"Based on only {len(labels_df)} labels — treat these as early signals, not "
            "settled conclusions. Confidence will improve with more labeling."
        )
        print()
        print(caveat)
        md_lines.append("")
        md_lines.append(f"*{caveat}*")

    TASTE_SUMMARY_PATH.write_text("\n".join(md_lines) + "\n")
    print()
    print(f"Saved taste summary to {TASTE_SUMMARY_PATH}")


def plot_removal_action_breakdown(labels_df: pd.DataFrame):
    """Bar chart of how many Removes were unsave-from-library vs
    remove-from-playlist vs remove-from-playlist-and-library vs the
    album fallback. Skips gracefully if no rows have this data yet
    (it was added after some labels were already collected)."""
    if "removal_action" not in labels_df.columns:
        print("No removal_action data yet (older labels predate this column) — skipping removal-action chart.")
        return

    counts = labels_df["removal_action"].dropna().value_counts()
    if counts.empty:
        print("No removal_action data yet (older labels predate this column) — skipping removal-action chart.")
        return

    print("=" * 60)
    print("REMOVAL ACTION BREAKDOWN")
    print("=" * 60)
    for action, count in counts.items():
        print(f"  {action:<30s} {count}")
    print()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values, color="#1DB954")
    plt.ylabel("Count")
    plt.title("Removal Action Breakdown")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REMOVAL_ACTION_CHART_PATH, dpi=150)
    plt.close()
    print(f"Saved removal action breakdown chart to {REMOVAL_ACTION_CHART_PATH}")


def plot_keep_rate_by_collection_type(labels_df: pd.DataFrame):
    """Bar chart of % kept per collection_type (liked/playlist/album). Skips
    gracefully if no rows have this data yet."""
    if "collection_type" not in labels_df.columns:
        print("No collection_type data yet (older labels predate this column) — skipping keep-rate-by-collection chart.")
        return

    subset = labels_df.dropna(subset=["collection_type"])
    if subset.empty:
        print("No collection_type data yet (older labels predate this column) — skipping keep-rate-by-collection chart.")
        return

    keep_rate = subset.groupby("collection_type")["label"].mean() * 100
    counts = subset.groupby("collection_type")["label"].count()

    print("=" * 60)
    print("KEEP RATE BY COLLECTION TYPE")
    print("=" * 60)
    for ctype, pct in keep_rate.items():
        print(f"  {ctype:<12s} {pct:5.1f}% kept  (n={counts[ctype]})")
    print()

    plt.figure(figsize=(7, 5))
    plt.bar(keep_rate.index.astype(str), keep_rate.values, color="#1DB954")
    plt.ylabel("% Kept")
    plt.ylim(0, 100)
    plt.title("Keep Rate by Collection Type")
    plt.tight_layout()
    plt.savefig(KEEP_RATE_BY_COLLECTION_CHART_PATH, dpi=150)
    plt.close()
    print(f"Saved keep-rate-by-collection chart to {KEEP_RATE_BY_COLLECTION_CHART_PATH}")


def plot_swipe_activity_over_time(labels_df: pd.DataFrame):
    """Line chart of swipes per day. Skips gracefully if swiped_at is missing
    or entirely null (older labels predate this column)."""
    if "swiped_at" not in labels_df.columns:
        print("No swiped_at data yet (older labels predate this column) — skipping swipe-activity chart.")
        return

    timestamps = pd.to_datetime(labels_df["swiped_at"], errors="coerce").dropna()
    if timestamps.empty:
        print("No swiped_at data yet (older labels predate this column) — skipping swipe-activity chart.")
        return

    daily_counts = timestamps.dt.date.value_counts().sort_index()

    print("=" * 60)
    print("SWIPE ACTIVITY OVER TIME")
    print("=" * 60)
    for day, count in daily_counts.items():
        print(f"  {day}: {count} swipes")
    print()

    plt.figure(figsize=(9, 5))
    plt.plot(daily_counts.index.astype(str), daily_counts.values, marker="o", color="#1DB954")
    plt.ylabel("Swipes")
    plt.title("Swipe Activity Over Time")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(SWIPE_ACTIVITY_CHART_PATH, dpi=150)
    plt.close()
    print(f"Saved swipe activity chart to {SWIPE_ACTIVITY_CHART_PATH}")


def main():
    if not LABELS_CSV_PATH.exists():
        print("data/labels.csv not found — swipe some tracks with `streamlit run app.py` first.")
        return
    labels_df, fills = load_labels_and_fills()
    if labels_df.empty:
        print("data/labels.csv is empty — nothing to evaluate.")
        return

    model, accuracy, baseline = evaluate_final_model(labels_df, fills)

    logreg_model, logreg_accuracy = evaluate_logistic_regression(labels_df, fills)
    print_model_comparison(accuracy, logreg_accuracy, baseline)
    plot_model_comparison(baseline, logreg_accuracy, accuracy)
    print()

    print(f"Simulating random-sampling baseline over {RANDOM_TRIALS} shuffles...")
    random_steps, random_acc = simulate_random_curve(labels_df, fills)

    if LEARNING_CURVE_CSV_PATH.exists():
        real_steps, real_acc = load_real_learning_curve()
    else:
        print("WARNING: data/learning_curve.csv not found — can't plot the real active-learning curve.")
        real_steps, real_acc = [], []

    if random_steps and real_steps:
        plot_comparison(real_steps, real_acc, random_steps, random_acc)
        print(f"Saved comparison chart to {COMPARISON_CHART_PATH}")
    else:
        print("WARNING: not enough k-fold-eligible points on one or both curves — skipping comparison chart.")
    print()

    sorted_names, sorted_importances = plot_feature_importance(model, FEATURE_COLUMNS)
    print(f"Saved feature importance chart to {FEATURE_IMPORTANCE_CHART_PATH}")

    print()
    print_logistic_regression_coefficients(logreg_model, FEATURE_COLUMNS)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total labels used: {len(labels_df)}")
    if accuracy is not None:
        lift_pts = (accuracy - baseline) * 100
        sign = "+" if lift_pts >= 0 else ""
        print(f"  Model accuracy:     {accuracy:.1%}")
        print(f"  Baseline accuracy:  {baseline:.1%}")
        print(f"  Lift over baseline: {sign}{lift_pts:.1f}pts")
    else:
        print("  Model accuracy: N/A (too few labels for k-fold CV)")
    print(f"  Top 3 features: {', '.join(sorted_names[:3])}")
    print("=" * 60)

    generate_taste_summary(labels_df, sorted_names, sorted_importances)

    print()
    plot_removal_action_breakdown(labels_df)
    print()
    plot_keep_rate_by_collection_type(labels_df)
    print()
    plot_swipe_activity_over_time(labels_df)


if __name__ == "__main__":
    main()
