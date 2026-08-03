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

RANDOM_TRIALS = 20


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

    return sorted_names


def main():
    if not LABELS_CSV_PATH.exists():
        print("data/labels.csv not found — swipe some tracks with `streamlit run app.py` first.")
        return
    labels_df, fills = load_labels_and_fills()
    if labels_df.empty:
        print("data/labels.csv is empty — nothing to evaluate.")
        return

    model, accuracy, baseline = evaluate_final_model(labels_df, fills)

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

    sorted_names = plot_feature_importance(model, FEATURE_COLUMNS)
    print(f"Saved feature importance chart to {FEATURE_IMPORTANCE_CHART_PATH}")

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


if __name__ == "__main__":
    main()
