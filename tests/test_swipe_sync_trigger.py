"""Drives the real app.render_swipe_session() (owner path) through simulated
swipes to verify the every-10-labels GitHub sync trigger and the manual
"Sync to GitHub now" sidebar button actually call commit_data_files_to_github()
at the right moments — without a browser, by mocking Streamlit's rendering
calls and running the real function body.

No real Spotify or GitHub calls: sp is a MagicMock, and
app.commit_data_files_to_github is patched to a spy.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import app


class _Rerun(Exception):
    """Stand-in for streamlit's internal RerunException — real streamlit
    halts script execution on st.rerun(); we just need to stop the call."""


def _make_tracks_df(n=30):
    return pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(n)],
            "track_name": [f"Track {i}" for i in range(n)],
            "artist_name": ["Artist"] * n,
            "album_name": ["Album"] * n,
            "image_url": [None] * n,
            "release_year": [2020] * n,
            "popularity": [50] * n,
            "days_since_added": [100] * n,
            "play_recency_days": [10] * n,
            "playlist_count": [1] * n,
            "artist_track_count": [5] * n,
        }
    )


class TestSwipeSyncTrigger(unittest.TestCase):
    def setUp(self):
        self.tracks_df = _make_tracks_df()
        st.session_state.clear()
        st.session_state.selected_collection = {
            "type": "liked",
            "id": None,
            "name": "All Liked Songs",
            "track_ids": self.tracks_df["track_id"].tolist(),
            "is_owned": True,
        }
        st.session_state.tracks_df = self.tracks_df
        st.session_state.labels_df = pd.DataFrame(
            columns=list(self.tracks_df.columns) + ["label"] + app.NEW_LABEL_COLUMNS
        )
        st.session_state.fills = app.compute_fill_values(self.tracks_df)
        st.session_state.sp = MagicMock()
        st.session_state.is_owner = True
        st.session_state.current_track_id = None
        st.session_state.last_accuracy = None
        st.session_state.labels_since_github_sync = 0
        st.session_state.spotify_display_name = "Test Owner"

    def _run_one_swipe(self, keep: bool, sync_button_clicked: bool = False):
        """Simulates one script run: streamlit re-executes render_swipe_session
        top to bottom, st.button returns True only for the clicked key."""

        def fake_button(label=None, key=None, **kwargs):
            if key == "sync_now_btn":
                return sync_button_clicked
            if sync_button_clicked:
                # Only one button is "clicked" per simulated script run, same
                # as real Streamlit — the sync click and a keep/remove click
                # can't both be true in the same run.
                return False
            if key == "keep_btn":
                return keep
            if key == "remove_btn":
                return not keep
            return False

        fake_col = MagicMock()
        fake_col.button.side_effect = fake_button
        fake_col.__enter__ = lambda s: s
        fake_col.__exit__ = lambda s, *a: False

        with patch.object(st, "button", side_effect=fake_button), \
             patch.object(st, "columns", return_value=(fake_col, fake_col)), \
             patch.object(st, "rerun", side_effect=_Rerun), \
             patch.object(st, "image"), patch.object(st, "toast"), \
             patch.object(st, "warning"), patch.object(st, "success"), \
             patch("app.append_csv_row"), \
             patch("app.LABELS_CSV_PATH", "/tmp/does-not-matter-labels.csv"), \
             patch("app.LEARNING_CURVE_CSV_PATH", "/tmp/does-not-matter-curve.csv"):
            # If sync button was clicked, app.py reads it inside the sidebar
            # block and calls commit_data_files_to_github() directly there,
            # separate from the keep/remove branch below.
            if sync_button_clicked:
                with patch("app.commit_data_files_to_github", return_value=True) as mock_commit:
                    try:
                        app.render_swipe_session()
                    except _Rerun:
                        pass
                    return mock_commit
            try:
                app.render_swipe_session()
            except _Rerun:
                pass
        return None

    @patch("app.commit_data_files_to_github")
    def test_sync_fires_on_10th_and_20th_label_not_others(self, mock_commit):
        mock_commit.return_value = True
        for i in range(1, 21):
            self._run_one_swipe(keep=True)
            n = st.session_state.labels_since_github_sync
            if i % 10 == 0:
                self.assertEqual(n, 0, f"counter should reset to 0 right after label {i}")
            else:
                self.assertEqual(n, i % 10, f"counter should be {i % 10} after label {i}")

        self.assertEqual(
            mock_commit.call_count, 2,
            f"expected exactly 2 syncs (after label 10 and label 20), got {mock_commit.call_count}",
        )
        self.assertEqual(len(st.session_state.labels_df), 20)

    def test_manual_sync_button_calls_commit_on_demand(self):
        # Not yet at the 10-label threshold — only the manual button should trigger it.
        for i in range(1, 4):
            self._run_one_swipe(keep=True)
        self.assertEqual(st.session_state.labels_since_github_sync, 3)

        mock_commit = self._run_one_swipe(keep=True, sync_button_clicked=True)
        mock_commit.assert_called_once()
        self.assertEqual(
            st.session_state.labels_since_github_sync, 0,
            "manual sync should reset the counter too, same as an automatic one",
        )


if __name__ == "__main__":
    unittest.main()
