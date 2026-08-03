"""Phase 5: LLM-generated "listener personality" summary and recommendation
directions.

Reads from data already on disk/in the Spotify API — tracks.csv, labels.csv,
evaluate.py's model/feature-importance logic, taste_summary.md — plus two new
cheap Spotify calls (top tracks/artists, already covered by the existing
user-top-read scope, no new auth needed). No Spotify writes, no changes to the
active-learning/model logic elsewhere in the project.

Two LLM providers, switched via the LLM_PROVIDER env var (default "anthropic"
— unset .env behaves exactly as before, no silent fallback between providers):
  - "anthropic" (default): the Claude API, ANTHROPIC_API_KEY.
  - "supportvectors": an OpenAI-compatible gateway (e.g. a bootcamp-provided
    endpoint), SUPPORTVECTORS_API_KEY + SUPPORTVECTORS_BASE_URL [+ _MODEL].
"""

from __future__ import annotations

import json
import os

import anthropic
from dotenv import load_dotenv

from evaluate import (
    FEATURE_COLUMNS,
    TASTE_SUMMARY_PATH,
    evaluate_final_model,
    load_labels_and_fills,
    plot_feature_importance,
)
from spotify_client import DATA_DIR, fetch_top_artists, fetch_top_tracks, get_spotify_client

LISTENER_SUMMARY_PATH = DATA_DIR / "listener_summary.md"
RECOMMENDATION_DIRECTIONS_PATH = DATA_DIR / "recommendation_directions.md"

ANTHROPIC_MODEL = "claude-sonnet-4-5"
SUPPORTVECTORS_DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1024
# Reasoning models need much more headroom — reasoning + answer share one budget.
SUPPORTVECTORS_MAX_TOKENS = 4096


def build_stats_digest() -> dict:
    """Pull together top tracks/artists, feature importances + label stats
    from evaluate.py's model logic, and the existing taste_summary.md into one
    structured dict to hand to Claude."""
    sp = get_spotify_client()

    top_tracks_raw = fetch_top_tracks(sp)
    top_artists_raw = fetch_top_artists(sp)

    top_tracks = [
        {
            "name": t.get("name"),
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
        }
        for t in top_tracks_raw
    ]
    # Artists don't have usable genre data via our API (same Extended Quota Mode
    # restriction as elsewhere in this project) — name only.
    top_artists = [a.get("name") for a in top_artists_raw if a.get("name")]

    labels_df, fills = load_labels_and_fills()

    label_stats = {"total_labels": 0, "keep_rate_pct": None}
    feature_importances = []

    if not labels_df.empty:
        model, accuracy, baseline = evaluate_final_model(labels_df, fills)
        sorted_names, sorted_importances = plot_feature_importance(model, FEATURE_COLUMNS)
        feature_importances = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(sorted_names, sorted_importances)
            if imp > 0
        ]
        label_stats = {
            "total_labels": len(labels_df),
            "keep_rate_pct": round(float(labels_df["label"].mean() * 100), 1),
            "model_accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
            "baseline_accuracy_pct": round(baseline * 100, 1),
        }

    taste_summary_text = TASTE_SUMMARY_PATH.read_text() if TASTE_SUMMARY_PATH.exists() else None

    return {
        "top_tracks": top_tracks,
        "top_artists": top_artists,
        "label_stats": label_stats,
        "feature_importances": feature_importances,
        "taste_summary": taste_summary_text,
    }


def _call_anthropic(prompt: str) -> str | None:
    """Never raises — returns None (with a printed warning) on any failure
    (missing key, auth, rate limit, API error, network), so a failed LLM call
    never crashes the caller or blocks the rest of the script."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "WARNING: ANTHROPIC_API_KEY not set — skipping LLM generation. "
            "Get one at https://console.anthropic.com/settings/keys and add it to .env."
        )
        return None

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        print("WARNING: Anthropic API authentication failed — check ANTHROPIC_API_KEY.")
        return None
    except anthropic.RateLimitError:
        print("WARNING: Anthropic API rate limited — try again shortly.")
        return None
    except anthropic.APIStatusError as e:
        print(f"WARNING: Anthropic API error ({e.status_code}): {e.message}")
        return None
    except anthropic.APIConnectionError as e:
        print(f"WARNING: couldn't reach the Anthropic API: {e}")
        return None

    return "".join(block.text for block in response.content if block.type == "text").strip()


def _call_supportvectors(prompt: str) -> str | None:
    """Never raises — returns None (with a printed warning) if config is
    missing or the request fails, same never-crash contract as _call_anthropic."""
    load_dotenv()
    api_key = os.getenv("SUPPORTVECTORS_API_KEY")
    base_url = os.getenv("SUPPORTVECTORS_BASE_URL")
    model = os.getenv("SUPPORTVECTORS_MODEL", SUPPORTVECTORS_DEFAULT_MODEL)

    missing = [
        name
        for name, val in (("SUPPORTVECTORS_API_KEY", api_key), ("SUPPORTVECTORS_BASE_URL", base_url))
        if not val
    ]
    if missing:
        print(
            f"WARNING: {', '.join(missing)} not set — skipping LLM generation "
            "(LLM_PROVIDER=supportvectors). Add them to .env."
        )
        return None

    from openai import OpenAI  # local import: only needed for this provider

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            # Reasoning-capable models (e.g. the default gpt-oss-20b) spend
            # tokens on an internal "reasoning" field before writing the final
            # answer, and both count against this same budget — a small budget
            # can be entirely consumed by reasoning, leaving empty content.
            max_tokens=SUPPORTVECTORS_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # A self-hosted/gateway endpoint's failure modes are less predictable
        # than Anthropic's typed exceptions (bad base URL, VPN not connected,
        # invalid model name, etc.), so this catches broadly and reports what
        # to check rather than crashing.
        print(f"WARNING: SupportVectors request failed: {e}")
        print(
            "  Check SUPPORTVECTORS_BASE_URL/API_KEY, that you're on the required "
            "network/VPN if applicable, and that SUPPORTVECTORS_MODEL is valid."
        )
        return None

    content = response.choices[0].message.content
    if not content:
        finish_reason = response.choices[0].finish_reason
        print(
            f"WARNING: SupportVectors returned empty content (finish_reason={finish_reason}). "
            "If this is a reasoning model, it likely spent the entire token budget on "
            "internal reasoning before writing an answer — try raising max_tokens further."
        )
        return None
    return content.strip()


def _call_llm(prompt: str) -> str | None:
    """Shared call path for both generation functions below. Dispatches on
    LLM_PROVIDER (default "anthropic") — never falls back silently from one
    provider to another; an unknown or misconfigured provider just skips with
    a printed warning."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        return _call_anthropic(prompt)
    if provider == "supportvectors":
        return _call_supportvectors(prompt)

    print(f"WARNING: unknown LLM_PROVIDER '{provider}' (expected 'anthropic' or 'supportvectors') — skipping.")
    return None


def generate_listener_summary(digest: dict) -> str | None:
    """Short second-person 'listener personality' writeup, grounded only in
    the digest — explicitly instructed not to invent facts."""
    prompt = (
        "Here is a JSON digest of someone's Spotify listening data and a small "
        "ML model trained on their Keep/Remove swipe decisions:\n\n"
        f"{json.dumps(digest, indent=2)}\n\n"
        "Write a short (150-200 word) \"listener personality\" summary in second "
        "person (\"you...\"). Base it ONLY on the stats provided above — do not "
        "invent facts, genres, artists, or listening habits that aren't present "
        "in this data. If the digest is sparse or missing in some area, don't "
        "speculate about it; just work with what's actually there."
    )
    return _call_llm(prompt)


def generate_recommendation_directions(digest: dict) -> str | None:
    """3-4 exploration DIRECTIONS (genre/era/artist-adjacency), not named
    tracks — we have no search step in this first version to verify a specific
    track recommendation actually exists on Spotify, so naming songs risks
    hallucinated titles. A search-verified track-recommendation step is a
    reasonable Phase 6 addition, not built here."""
    prompt = (
        "Here is a JSON digest of someone's Spotify listening data and taste "
        "patterns learned from their Keep/Remove swipe decisions:\n\n"
        f"{json.dumps(digest, indent=2)}\n\n"
        "Suggest 3-4 DIRECTIONS for what to explore next — NOT specific song "
        "titles or named tracks (there is no way to verify a specific track "
        "recommendation actually exists, so don't name songs). Instead suggest "
        "things like genres to try, eras/decades to explore, or \"artists "
        "adjacent to X\" style directions. For each, give a short reason "
        "grounded in the taste patterns in the digest above (e.g. artist "
        "loyalty, era preference, recency bias). Do not invent facts not "
        "present in the digest."
    )
    return _call_llm(prompt)


def main():
    print("Building stats digest from Spotify + local data...")
    digest = build_stats_digest()

    print()
    print("=" * 60)
    print("LISTENER PERSONALITY SUMMARY")
    print("=" * 60)
    summary = generate_listener_summary(digest)
    if summary:
        print(summary)
        LISTENER_SUMMARY_PATH.write_text(summary + "\n")
        print(f"\nSaved to {LISTENER_SUMMARY_PATH}")
    else:
        print("Skipped — see warning above.")

    print()
    print("=" * 60)
    print("RECOMMENDATION DIRECTIONS")
    print("=" * 60)
    directions = generate_recommendation_directions(digest)
    if directions:
        print(directions)
        RECOMMENDATION_DIRECTIONS_PATH.write_text(directions + "\n")
        print(f"\nSaved to {RECOMMENDATION_DIRECTIONS_PATH}")
    else:
        print("Skipped — see warning above.")


if __name__ == "__main__":
    main()
