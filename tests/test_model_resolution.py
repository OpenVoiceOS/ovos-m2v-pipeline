"""``_resolve_model_revision`` must validate a Hub repo id against the Hub on
every load when no ``revision`` is pinned. model2vec's own resolution
(``maybe_get_cached_model_path``) returns the newest snapshot already on
disk BY MTIME with no Hub contact at all, so a stale pre-relabel snapshot
gets loaded forever once one is cached -- this is exactly what happened with
OpenVoiceOS/ovos-m2v-intents-multi-128M-v5 (see docs/configuration.md).

With no revision pinned, that Hub check must also never block boot for
minutes on a slow or unreachable Hub -- ``_cached_snapshot_if_current``
bounds it to one cheap, short-timeout ``HfApi.model_info`` call instead of
a full per-file ``snapshot_download`` etag round trip.
"""
import unittest
from unittest.mock import MagicMock, patch

import huggingface_hub
import requests
from ovos_bus_client.message import Message  # noqa: F401  (import parity with sibling tests)
from ovos_utils.fakebus import FakeBus


def _make_pipeline(config):
    from ovos_m2v_pipeline import Model2VecIntentPipeline
    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}):
        return Model2VecIntentPipeline(bus=FakeBus(), config=config)


class TestResolveModelRevision(unittest.TestCase):
    def test_revision_unset_validates_against_hub(self):
        """No pin, empty cache: still resolved through `snapshot_download`
        (no revision kwarg value) so a stale local snapshot never wins over
        the Hub."""
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None) as mock_cached, \
             patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/deadbeef") as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")
        mock_cached.assert_called_once_with("OpenVoiceOS/fake-repo")
        mock_dl.assert_called_once_with(
            "OpenVoiceOS/fake-repo", repo_type="model", revision=None)
        self.assertEqual(result, "/cache/fake-repo/snapshots/deadbeef")

    def test_revision_pinned_is_passed_through(self):
        pipeline = _make_pipeline({
            "model": "OpenVoiceOS/fake-repo", "revision": "v5"})
        with patch.object(pipeline, "_cached_snapshot_if_current") as mock_cached, \
             patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/v5") as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")
        mock_cached.assert_not_called()  # pinned commits are immutable, no probe needed
        mock_dl.assert_called_once_with(
            "OpenVoiceOS/fake-repo", repo_type="model", revision="v5")
        self.assertEqual(result, "/cache/fake-repo/snapshots/v5")

    def test_local_directory_is_returned_unchanged(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("ovos_m2v_pipeline.Path.exists", return_value=True), \
             patch("huggingface_hub.snapshot_download") as mock_dl:
            result = pipeline._resolve_model_revision("/some/local/dir")
        mock_dl.assert_not_called()
        self.assertEqual(result, "/some/local/dir")

    def test_hub_unreachable_falls_back_to_cache_with_warning(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   side_effect=[requests.exceptions.ConnectionError("no net"),
                                "/cache/fake-repo/snapshots/stale"]) as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False), \
             patch("ovos_m2v_pipeline.LOG") as mock_log:
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")

        self.assertEqual(mock_dl.call_count, 2)
        first_kwargs = mock_dl.call_args_list[0].kwargs
        self.assertNotIn("local_files_only", first_kwargs)
        second_kwargs = mock_dl.call_args_list[1].kwargs
        self.assertTrue(second_kwargs.get("local_files_only"))
        self.assertEqual(result, "/cache/fake-repo/snapshots/stale")
        mock_log.warning.assert_called_once()
        self.assertIn("/cache/fake-repo/snapshots/stale",
                      mock_log.warning.call_args[0][0])

    def test_hub_http_error_falls_back_to_cache(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        http_error = huggingface_hub.utils.HfHubHTTPError(
            "500 error", response=MagicMock())
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   side_effect=[http_error,
                                "/cache/fake-repo/snapshots/stale"]) as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False), \
             patch("ovos_m2v_pipeline.LOG"):
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")
        self.assertEqual(mock_dl.call_count, 2)
        self.assertEqual(result, "/cache/fake-repo/snapshots/stale")

    def test_offline_env_var_uses_cache_directly_with_warning(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/offline") as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), \
             patch("ovos_m2v_pipeline.LOG") as mock_log:
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")

        mock_dl.assert_called_once()
        self.assertTrue(mock_dl.call_args.kwargs.get("local_files_only"))
        self.assertEqual(result, "/cache/fake-repo/snapshots/offline")
        mock_log.warning.assert_called_once()
        self.assertIn("HF_HUB_OFFLINE", mock_log.warning.call_args[0][0])


class TestCachedSnapshotIfCurrent(unittest.TestCase):
    """``_cached_snapshot_if_current`` is the bounded probe that lets an
    unpinned repo id skip a full `snapshot_download` etag round trip when
    the cache already matches the Hub's current commit."""

    def test_cache_matches_hub_skips_full_download(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/abc123") as mock_dl, \
             patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.model_info.return_value = MagicMock(sha="abc123")
            result = pipeline._cached_snapshot_if_current("OpenVoiceOS/fake-repo")

        mock_dl.assert_called_once_with(
            "OpenVoiceOS/fake-repo", repo_type="model", local_files_only=True)
        mock_api.return_value.model_info.assert_called_once_with(
            "OpenVoiceOS/fake-repo", timeout=5.0)
        self.assertEqual(result, "/cache/fake-repo/snapshots/abc123")

    def test_cache_stale_returns_none(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/abc123"), \
             patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.model_info.return_value = MagicMock(sha="deadbeef")
            result = pipeline._cached_snapshot_if_current("OpenVoiceOS/fake-repo")
        self.assertIsNone(result)

    def test_nothing_cached_returns_none_without_probing_hub(self):
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("huggingface_hub.snapshot_download",
                   side_effect=huggingface_hub.utils.LocalEntryNotFoundError("no cache")), \
             patch("huggingface_hub.HfApi") as mock_api:
            result = pipeline._cached_snapshot_if_current("OpenVoiceOS/fake-repo")
        mock_api.return_value.model_info.assert_not_called()
        self.assertIsNone(result)

    def test_probe_timeout_falls_back_to_cache_without_downloading(self):
        """The whole point of the bound: a blackholed or slow Hub must
        raise (a timeout, a connection error) rather than hang, and that
        failure must resolve to the cached snapshot, not a real download.
        """
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch("huggingface_hub.snapshot_download",
                   return_value="/cache/fake-repo/snapshots/abc123") as mock_dl, \
             patch("huggingface_hub.HfApi") as mock_api, \
             patch("ovos_m2v_pipeline.LOG") as mock_log:
            mock_api.return_value.model_info.side_effect = requests.exceptions.ConnectTimeout(
                "connect timed out")
            result = pipeline._cached_snapshot_if_current("OpenVoiceOS/fake-repo")

        mock_dl.assert_called_once_with(
            "OpenVoiceOS/fake-repo", repo_type="model", local_files_only=True)
        self.assertEqual(result, "/cache/fake-repo/snapshots/abc123")
        mock_log.warning.assert_called_once()
        self.assertIn("/cache/fake-repo/snapshots/abc123",
                      mock_log.warning.call_args[0][0])

    def test_resolve_model_revision_uses_probe_result_without_full_download(self):
        """End-to-end: when the probe confirms the cache is current,
        `_resolve_model_revision` must not also do a full online
        `snapshot_download` -- that would defeat the whole bound."""
        pipeline = _make_pipeline({"model": "OpenVoiceOS/fake-repo"})
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value="/cache/fake-repo/snapshots/abc123") as mock_cached, \
             patch("huggingface_hub.snapshot_download") as mock_dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision("OpenVoiceOS/fake-repo")
        mock_cached.assert_called_once_with("OpenVoiceOS/fake-repo")
        mock_dl.assert_not_called()
        self.assertEqual(result, "/cache/fake-repo/snapshots/abc123")


if __name__ == "__main__":
    unittest.main()
