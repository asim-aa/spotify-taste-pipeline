"""Mock-only tests for github_sync.commit_data_files_to_github. Never hits
the real GitHub API — every requests.get/put call is patched. Runs against a
temp directory, never the project's real data/ files."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import github_sync


def _resp(status_code, json_body=None, text=""):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_body or {}
    m.text = text
    return m


class TestCommitDataFilesToGithub(unittest.TestCase):
    def setUp(self):
        github_sync._warned_missing_token = False
        self._tmpdir = tempfile.TemporaryDirectory()
        self._real_data_dir = github_sync.DATA_DIR
        github_sync.DATA_DIR = Path(self._tmpdir.name)

    def tearDown(self):
        github_sync.DATA_DIR = self._real_data_dir
        self._tmpdir.cleanup()

    def _make_data_files(self):
        for name in ("labels.csv", "learning_curve.csv"):
            (github_sync.DATA_DIR / name).write_text("track_id,label\nabc123,1\n")

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token", "GITHUB_REPO": "user/repo"})
    @patch("github_sync.requests.put")
    @patch("github_sync.requests.get")
    def test_successful_round_trip(self, mock_get, mock_put):
        self._make_data_files()
        mock_get.return_value = _resp(200, {"sha": "abc123sha"})
        mock_put.return_value = _resp(200)

        result = github_sync.commit_data_files_to_github()

        self.assertTrue(result)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_put.call_count, 2)
        # SHA from the GET must be threaded into the PUT payload.
        for call in mock_put.call_args_list:
            self.assertEqual(call.kwargs["json"]["sha"], "abc123sha")

    @patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token", "GITHUB_REPO": "user/repo"})
    @patch("github_sync.requests.put")
    @patch("github_sync.requests.get")
    def test_409_conflict_warns_and_skips_without_clobbering(self, mock_get, mock_put):
        self._make_data_files()
        mock_get.return_value = _resp(200, {"sha": "stale-sha"})
        mock_put.return_value = _resp(409, text="sha mismatch")

        with patch("builtins.print") as mock_print:
            result = github_sync.commit_data_files_to_github()

        self.assertFalse(result)
        # Every PUT was attempted (skip means "don't retry with a new SHA",
        # not "don't attempt at all") but none should be treated as success.
        self.assertEqual(mock_put.call_count, 2)
        warned = any("Conflict" in str(c.args) for c in mock_print.call_args_list)
        self.assertTrue(warned, "expected a printed conflict warning")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_token_skips_gracefully_and_warns_once(self):
        with patch("builtins.print") as mock_print:
            result1 = github_sync.commit_data_files_to_github()
            result2 = github_sync.commit_data_files_to_github()

        self.assertFalse(result1)
        self.assertFalse(result2)
        warnings = [c for c in mock_print.call_args_list if "GITHUB_TOKEN" in str(c.args)]
        self.assertEqual(len(warnings), 1, "expected the missing-token message exactly once, not per call")


if __name__ == "__main__":
    unittest.main()
