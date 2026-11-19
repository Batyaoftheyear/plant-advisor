"""Check that export packages weights and inputs without optimizer checkpoints."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_submission_artifacts import SFT_PATH, make_bundle


class ArtifactExportTests(unittest.TestCase):
    def setUp(self):
        self.cache = ROOT / ".cache/export-tests"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.cache)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.cache.resolve()))
        self.inputs = self.root / "inputs"
        self.run = self.root / "run"
        manifest = json.loads((ROOT / "data/v3/manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"] + [{"path": "data/v3/manifest.json"}]:
            target = self.inputs / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / entry["path"], target)
        sft = self.run / SFT_PATH
        for name in ["environment.json", "training_log.csv", "train_metrics.json",
                     "best_checkpoint.json", "loss_curves.png", "adapter/adapter_config.json",
                     "adapter/adapter_model.safetensors", "checkpoints/checkpoint-15/optimizer.pt"]:
            path = sft / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"export-test-fixture")
        training = {"data_sha256": {
            key: hashlib.sha256((self.inputs / f"data/v3/{key}.jsonl").read_bytes()).hexdigest()
            for key in ["train", "validation"]
        }}
        (sft / "training_config.json").write_text(json.dumps(training), encoding="utf-8")
        self.archive = self.root / "submission.zip"

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.cache.resolve()))
        self.assertNotEqual(self.root, self.cache.resolve())
        self.temporary.cleanup()

    def test_training_bundle_preserves_file_bytes_and_omits_checkpoints(self):
        result = make_bundle(self.run, self.inputs, self.archive)
        self.assertEqual(result["sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        with zipfile.ZipFile(self.archive) as archive:
            self.assertFalse(any("checkpoints/" in name for name in archive.namelist()))
            self.assertIn((SFT_PATH / "adapter/adapter_model.safetensors").as_posix(), archive.namelist())
            manifest = json.loads(archive.read("ARTIFACTS_MANIFEST.json"))
            self.assertEqual(manifest["stage"], "training")
            for entry in manifest["files"]:
                data = archive.read(entry["path"])
                self.assertEqual(entry["size"], len(data))
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())

    def test_missing_weights_prevent_incomplete_export(self):
        (self.run / SFT_PATH / "adapter/adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            make_bundle(self.run, self.inputs, self.archive)
        self.assertFalse(self.archive.exists())

    def test_missing_evaluation_prevents_claiming_complete_bundle(self):
        with self.assertRaises(FileNotFoundError):
            make_bundle(self.run, self.inputs, self.archive, include_evaluation=True)
        self.assertFalse(self.archive.exists())

    def test_changed_training_data_prevent_export(self):
        (self.inputs / "data/v3/train.jsonl").write_bytes(b"different dataset")
        with self.assertRaisesRegex(ValueError, "данные"):
            make_bundle(self.run, self.inputs, self.archive)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
