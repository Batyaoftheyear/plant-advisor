"""Synchronization tests: version isolation, complete downloads and integrity guards."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import sync_colab_data as sync


class ColabSyncTests(unittest.TestCase):
    def setUp(self):
        self.cache = REPOSITORY_ROOT / ".cache" / "sync-tests"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.cache)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.cache.resolve()))
        self.manifest = json.loads((REPOSITORY_ROOT / sync.MANIFEST_PATH).read_text(encoding="utf-8"))
        self.remote = {item["path"]: (REPOSITORY_ROOT / item["path"]).read_bytes()
                       for item in self.manifest["files"]}
        self.remote[sync.MANIFEST_PATH] = json.dumps(self.manifest).encode()
        self.commit = "a" * 40
        self.urls = []
        # Simulate old Drive contents that must never be overwritten by v3.
        self.legacy = self.root / "data/train.jsonl"
        self.legacy.parent.mkdir(parents=True)
        self.legacy.write_bytes(b"previous-v2-data")
        self.old_adapter = self.root / "results/sft/qwen25_3b_qlora_v2/adapter/sentinel"
        self.old_adapter.parent.mkdir(parents=True)
        self.old_adapter.write_bytes(b"previous-v2-adapter")

    def tearDown(self):
        # Check the resolved path before TemporaryDirectory recursively removes it.
        self.assertTrue(self.root.is_relative_to(self.cache.resolve()))
        self.assertNotEqual(self.root, self.cache.resolve())
        self.temporary.cleanup()

    def urlopen(self, request, timeout):
        self.assertEqual(timeout, 60)
        url = request.full_url
        self.urls.append(url)
        prefix = f"https://raw.githubusercontent.com/{sync.DEFAULT_REPOSITORY}/{self.commit}/"
        self.assertTrue(url.startswith(prefix), url)
        return io.BytesIO(self.remote[url.removeprefix(prefix)])

    def synchronize(self):
        resolved = subprocess.CompletedProcess([], 0, stdout=self.commit + "\trefs/heads/main\n")
        with patch.object(sync, "subprocess") as git, patch.object(sync, "urlopen", side_effect=self.urlopen):
            git.run.return_value = resolved
            return sync.sync_from_github(self.root)

    def assert_legacy_unchanged(self):
        self.assertEqual(self.legacy.read_bytes(), b"previous-v2-data")
        self.assertEqual(self.old_adapter.read_bytes(), b"previous-v2-adapter")

    def test_initial_sync_pins_one_commit_and_preserves_v2(self):
        receipt = self.synchronize()
        self.assertEqual(receipt["github_commit"], self.commit)
        self.assertEqual(len(receipt["updated"]), 12)
        self.assertEqual(sync.verify_local(self.root)["dataset_version"], "v3")
        self.assertEqual(len(self.urls), 13)  # manifest + 12 files from the pinned commit
        self.assert_legacy_unchanged()

    def test_repeated_sync_does_not_download_unchanged_files(self):
        self.synchronize()
        self.urls.clear()
        receipt = self.synchronize()
        self.assertEqual(receipt["updated"], [])
        self.assertEqual(len(receipt["unchanged"]), 12)
        self.assertEqual(len(self.urls), 1)
        self.assertFalse((self.root / "sync_backups").exists())
        self.assert_legacy_unchanged()

    def test_bad_download_never_partially_installs_a_dataset(self):
        self.remote["data/v3/train.jsonl"] = b"truncated download"
        with self.assertRaisesRegex(ValueError, "контрольная сумма"):
            self.synchronize()
        self.assertFalse((self.root / "data/v3").exists())
        self.assertFalse((self.root / "results/baseline").exists())
        self.assert_legacy_unchanged()

    def test_conflicting_shared_baseline_is_not_overwritten(self):
        shared = self.root / "results/baseline/qwen25_3b_512/config.json"
        shared.parent.mkdir(parents=True)
        shared.write_bytes(b"different experiment")
        with self.assertRaisesRegex(ValueError, "Общий файл"):
            self.synchronize()
        self.assertEqual(shared.read_bytes(), b"different experiment")
        self.assertFalse((self.root / "data/v3").exists())
        self.assert_legacy_unchanged()

    def test_changed_v3_file_is_backed_up_before_update(self):
        changed = self.root / "data/v3/train.jsonl"
        changed.parent.mkdir(parents=True)
        changed.write_bytes(b"older v3 content")
        receipt = self.synchronize()
        backup = Path(receipt["backup_directory"]) / "data/v3/train.jsonl"
        self.assertEqual(backup.read_bytes(), b"older v3 content")
        self.assertEqual(changed.read_bytes(), self.remote["data/v3/train.jsonl"])
        self.assertIn("data/v3/train.jsonl", receipt["backed_up"])
        self.assert_legacy_unchanged()

    def test_unexpected_manifest_path_is_rejected_before_writes(self):
        self.manifest["files"][0]["path"] = "../outside.jsonl"
        self.remote[sync.MANIFEST_PATH] = json.dumps(self.manifest).encode()
        with self.assertRaises(ValueError):
            self.synchronize()
        self.assertFalse((self.root / "data/v3").exists())
        self.assert_legacy_unchanged()

    def test_local_verification_detects_changes_after_sync(self):
        self.synchronize()
        changed = self.root / "data/v3/train.jsonl"
        changed.write_bytes(b"manual modification")
        with self.assertRaisesRegex(ValueError, "контрольная сумма"):
            sync.verify_local(self.root)
        self.assertEqual(changed.read_bytes(), b"manual modification")


if __name__ == "__main__":
    unittest.main()
