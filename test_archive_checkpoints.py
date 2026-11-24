"""Offline regression tests for backup from an installed Hugging Face cache."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools import archive_checkpoints as backup


REVISION = "a" * 40


def make_fixture(root):
    cache = root / "huggingface"
    data = root / "data.zip"
    output = root / "complete.zip"
    payload = b"source data"
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data/sample", payload)
        metadata = {"files": {"data/sample": hashlib.sha256(payload).hexdigest()}}
        archive.writestr(backup.DATA_MANIFEST, json.dumps(metadata))
    snapshots = {}
    for model in backup.MODELS:
        repo = cache / "hub" / ("models--" + model.replace("/", "--"))
        snapshot = repo / "snapshots" / REVISION
        snapshots[model] = snapshot
        snapshot.mkdir(parents=True)
        for name, content in [("config.json", {"model_type": "bert"}),
                              ("tokenizer_config.json", {}), ("tokenizer.json", {})]:
            (snapshot / name).write_text(json.dumps(content))
        if model == backup.ENCODER:
            (snapshot / "modules.json").write_text('[{"path": "1_Pooling"}]')
            (snapshot / "1_Pooling").mkdir()
            (snapshot / "1_Pooling/config.json").write_text('{"pooling_mode_mean_tokens": true}')
        # Two small valid safetensors shards with one tensor each exercise indexed weights.
        shards = {}
        for i in (1, 2):
            shard = f"model-{i:05}-of-00002.safetensors"
            key = f"layer_{i}"
            header = json.dumps({key: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
            (snapshot / shard).write_bytes(len(header).to_bytes(8, "little") + header + bytes([i]))
            shards[key] = shard
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shards}))
        (snapshot / "unrelated-private-file").write_text("exclude")
    (cache / "token").write_text("exclude")
    return data, cache, output, snapshots


class CheckpointBackupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data, self.cache, self.output, self.snapshots = make_fixture(Path(temp.name))

    def test_restores_regular_files_and_only_explicit_revision(self):
        snapshot = self.snapshots[backup.MODELS[0]]
        (snapshot / "tokenizer_config.json").write_text(json.dumps({
            "normalizer_file": "normalization_rules.json", "auto_map": ["tokenization_example.Tokenizer", None],
        }))
        (snapshot / "normalization_rules.json").write_text("{}")
        (snapshot / "tokenization_example.py").write_text("# cached optional tokenizer")
        (snapshot.parent / ("b" * 40)).mkdir()
        with self.assertRaisesRegex(ValueError, "cached revisions"):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertFalse(self.output.exists())
        backup.extend_archive(self.data, self.cache, self.output, {backup.MODELS[0]: REVISION})
        with zipfile.ZipFile(self.output) as archive:
            backup.verify_zip(archive, with_models=True)
            names = archive.namelist()
            self.assertFalse(any("unrelated" in name or name.endswith("/token") for name in names))
            self.assertEqual(8, sum(n.endswith(".safetensors") for n in names))
            self.assertTrue(any(n.endswith("/normalization_rules.json") for n in names))
            self.assertTrue(any(n.endswith("/tokenization_example.py") for n in names))
            self.assertEqual(archive.read("data/sample"), b"source data")
            for model in backup.MODELS:
                prefix = "checkpoints/huggingface/hub/models--" + model.replace("/", "--")
                self.assertEqual(REVISION, archive.read(prefix + "/refs/main").decode())

    def test_invalid_shards_do_not_create_output(self):
        shard = self.snapshots[backup.ENCODER] / "model-00002-of-00002.safetensors"
        original = shard.read_bytes()
        shard.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertFalse(self.output.exists())
        shard.unlink()
        with self.assertRaises(FileNotFoundError):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertFalse(self.output.exists())
        index = shard.with_name("model.safetensors.index.json")
        mapping = json.loads(index.read_text())
        del mapping["weight_map"]["layer_2"]
        index.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, "Missing shard references"):
            backup.extend_archive(self.data, self.cache, self.output, {})

    def test_source_manifest_failure_leaves_no_output_and_existing_output_is_preserved(self):
        with zipfile.ZipFile(self.data, "w") as archive:
            archive.writestr("data/sample", b"modified data")
            archive.writestr(backup.DATA_MANIFEST, json.dumps({"files": {"data/sample": "f" * 64}}))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertFalse(self.output.exists())
        self.output.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_shared_blob_uses_repository_digest_and_rejects_external_target(self):
        snapshot = self.snapshots[backup.MODELS[0]]
        shard = snapshot / "model-00001-of-00002.safetensors"
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        repo = snapshot.parent.parent
        blob = repo / "blobs" / digest
        blob.parent.mkdir()
        shared = repo.parent / "blobs" / "7f" / ("7f" * 32)
        shared.parent.mkdir(parents=True)
        shard.rename(shared)
        try:
            shard.symlink_to(Path("..") / ".." / "blobs" / digest)
            blob.symlink_to(Path("..") / ".." / "blobs" / "7f" / shared.name)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        backup.extend_archive(self.data, self.cache, self.output, {})
        with zipfile.ZipFile(self.output) as archive:
            backup.verify_zip(archive, with_models=True)
        blob.unlink()
        external = self.cache / shared.name
        shared.rename(external)
        blob.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Unexpected shared blob target"):
            backup.cached_file(snapshot, shard.name)

    def test_bad_blob_digest_rejects_complete_but_substituted_shard(self):
        snapshot = self.snapshots[backup.ENCODER]
        shard = snapshot / "model-00002-of-00002.safetensors"
        blob = snapshot.parent.parent / "blobs" / ("f" * 64)
        blob.parent.mkdir()
        shard.rename(blob)
        try:
            shard.symlink_to(blob)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "blob SHA-256 mismatch"):
            backup.extend_archive(self.data, self.cache, self.output, {})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
