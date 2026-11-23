"""Add the four cached benchmark checkpoints to a verified data ZIP, without downloads.

The restored Hugging Face cache contains regular snapshot files and refs/main.
This is the same layout used by the Hub when filesystem symlinks are unavailable.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODELS = ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3.5-4B", "LiquidAI/d1-3B", ENCODER)
DATA_MANIFEST = "data/archive-manifest.json"
MODEL_MANIFEST = "checkpoints/manifest.json"
COMMIT = re.compile(r"[0-9a-f]{40}")
CHUNK = 4 * 1024 * 1024
# Runtime assets only. In particular, never walk the user-level Hub configuration.
ASSETS = (
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "vocab.txt",
    "tokenizer.model", "spiece.model", "normalizer.json", "tokenizer.tiktoken",
    "chat_template.json", "chat_template.jinja", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "modules.json", "1_Pooling/config.json", "README.md", "LICENSE",
)


def transfer(source, dest=None):
    """Hash a file in bounded memory, optionally copying it to an open ZIP member."""
    digest = hashlib.sha256()
    for part in iter(lambda: source.read(CHUNK), b""):
        digest.update(part)
        if dest is not None:
            dest.write(part)
    return digest.hexdigest()


def verify_zip(archive, with_models=None):
    if with_models is None:
        with_models = MODEL_MANIFEST in archive.namelist()
    manifests = [DATA_MANIFEST, MODEL_MANIFEST] if with_models else [DATA_MANIFEST]
    expected = {}
    for name in manifests:
        record = json.loads(archive.read(name))
        entries = record["files"]
        if name == MODEL_MANIFEST:
            models = record["models"]
            if set(models) != set(MODELS):
                raise ValueError("Missing or unexpected models in checkpoint manifest")
            for model, revision in models.items():
                prefix = f"checkpoints/huggingface/hub/models--{model.replace('/', '--')}"
                if (not COMMIT.fullmatch(revision) or archive.read(prefix + "/refs/main") != revision.encode()
                        or f"{prefix}/snapshots/{revision}/config.json" not in entries):
                    raise ValueError(f"Invalid snapshot reference for {model}")
        roots = {"checkpoints"} if name == MODEL_MANIFEST else {"data", "results"}
        for path, digest in entries.items():
            parts = path.split("/")
            if (parts[0] not in roots or any(p in {"", ".", ".."} for p in parts)
                    or "\\" in path or ":" in path or path in manifests or path in expected
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError(f"Invalid entry in {name}: {path}")
            expected[path] = digest
    members = archive.namelist()
    if len(set(members)) != len(members) or set(members) != set(manifests) | expected.keys():
        raise ValueError("The ZIP contents differ from the listed files in its manifest(s)")
    for name, digest in expected.items():
        with archive.open(name) as stream:
            if transfer(stream) != digest:
                raise ValueError(f"SHA-256 mismatch for {name}")
    return with_models


def find_revision(repo, model, revisions):
    snapshots = repo / "snapshots"
    chosen = revisions.get(model)
    if chosen is None:
        choices = sorted(p.name for p in snapshots.iterdir() if p.is_dir() and COMMIT.fullmatch(p.name))
        if len(choices) != 1:
            raise ValueError(f"{model} has {len(choices)} cached revisions ({', '.join(choices)}). "
                             "Select the experimental commit with --revision MODEL=SHA")
        chosen = choices[0]
    snapshot = snapshots / chosen
    if not snapshot.is_dir() or snapshot.resolve() != repo.resolve() / "snapshots" / chosen:
        raise ValueError(f"Missing or redirected snapshot for {model} at {chosen}")
    return chosen, snapshot


def cached_file(snapshot, name):
    path = snapshot / name
    resolved = path.resolve(strict=True)
    blob_dir = snapshot.parent.parent / "blobs"
    if resolved != snapshot.resolve() / name:
        if resolved.parent != blob_dir or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name):
            raise ValueError(f"Unexpected symlink target for {path}")
    if not resolved.is_file() or not resolved.stat().st_size:
        raise ValueError(f"Missing or empty model file: {path}")
    return resolved


def safe_tensor_keys(path):
    """Check the format length and offsets so truncated shards cannot pass preflight."""
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 2 <= size <= 100_000_000:
            raise ValueError(f"Invalid safetensors header in {path}")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    offsets = sorted(t["data_offsets"] for t in header.values())
    end = 0
    for first, last in offsets:
        if first != end or last < first:
            raise ValueError(f"Inconsistent tensor offsets in {path}")
        end = last
    if not header or path.stat().st_size != 8 + size + end:
        raise ValueError(f"Truncated or inconsistent safetensors file: {path}")
    return header.keys()


def model_files(snapshot, model):
    files = {n: cached_file(snapshot, n) for n in ASSETS if (snapshot / n).exists()}
    configs = {n: json.loads(file.read_bytes()) for n, file in files.items() if n.endswith(".json")}
    if "model_type" not in configs.get("config.json", {}):
        raise ValueError(f"Missing model_type in {model}/config.json")
    if "tokenizer_config.json" not in files or not (
        {"tokenizer.json", "tokenizer.model", "vocab.txt"} & files.keys()
        or {"vocab.json", "merges.txt"} <= files.keys()
    ):
        raise ValueError(f"Incomplete tokenizer in {model}")
    if model == ENCODER and (not configs.get("modules.json") or not configs.get("1_Pooling/config.json")):
        raise ValueError(f"Missing sentence-transformer modules or pooling configuration in {model}")
    # Preserve additional tokenizer resources or model modules explicitly referenced by
    # cached configs. Restrict resolution to asset names inside the known repo snapshot.
    for config in (configs.get(name, {}) for name in ("config.json", "tokenizer_config.json", "processor_config.json")):
        for key in ("vocab_file", "merges_file", "tokenizer_file", "normalizer_file", "sp_model_file"):
            name = config.get(key)
            if isinstance(name, str) and re.fullmatch(r"[\w.-]+\.(json|txt|model|tiktoken)", name):
                files[name] = cached_file(snapshot, name)
        auto_map = config.get("auto_map") or {}
        # Older tokenizer configs stored a list for the slow/fast implementations.
        entries = auto_map.values() if isinstance(auto_map, dict) else [auto_map]
        for entry in entries:
            for reference in (entry if isinstance(entry, list) else [entry]):
                if isinstance(reference, str) and re.fullmatch(r"\w+\.\w+", reference):
                    name = reference.split(".")[0] + ".py"
                    if (snapshot / name).exists():
                        files[name] = cached_file(snapshot, name)
    for template in (snapshot / "chat_templates").glob("*.jinja"):
        name = "chat_templates/" + template.name
        files[name] = cached_file(snapshot, name)
    # Prefer the format loaded by the standard inference clients. Keep only one weight set.
    for weights, pattern in [("model.safetensors", r"model-(\d+)-of-(\d+)\.safetensors"),
                             ("pytorch_model.bin", r"pytorch_model-(\d+)-of-(\d+)\.bin")]:
        index = weights + ".index.json"
        if (snapshot / index).exists():
            files[index] = cached_file(snapshot, index)
            mapping = json.loads(files[index].read_bytes())["weight_map"]
            if not mapping or not all(re.fullmatch(pattern, shard) for shard in mapping.values()):
                raise ValueError(f"Invalid weight map in {model}/{index}")
            shards = set(mapping.values())
            numbering = {tuple(map(int, re.fullmatch(pattern, shard).groups())) for shard in shards}
            if numbering != {(i, len(shards)) for i in range(1, len(shards) + 1)}:
                raise ValueError(f"Missing shard references in {model}/{index}")
        elif (snapshot / weights).exists():
            mapping, shards = {}, {weights}
        else:
            continue
        for shard in sorted(shards):
            path = cached_file(snapshot, shard)
            if shard.endswith(".safetensors"):
                keys = safe_tensor_keys(path)
                if any(key not in keys for key, file in mapping.items() if file == shard):
                    raise ValueError(f"Indexed tensors missing from {model}/{shard}")
            else:
                # Modern torch.save weights are ZIPs. Inspect their internal structure
                # without importing torch or executing the pickle inside the file.
                with zipfile.ZipFile(path) as weights_zip:
                    members = weights_zip.namelist()
                    if not any(n.endswith("/data.pkl") for n in members) or not any("/data/" in n for n in members):
                        raise ValueError(f"Invalid PyTorch weight file in {model}/{shard}")
            files[shard] = path
        return files
    raise ValueError(f"No complete supported weight set for {model} in {snapshot}")


def extend_archive(source, cache, output, revisions):
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new ZIP path")
    plans = {}
    for model in MODELS:
        repo = (cache / "hub" / ("models--" + model.replace("/", "--"))).resolve()
        revision, snapshot = find_revision(repo, model, revisions)
        plans[model] = (repo.name, revision, model_files(snapshot, model))
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".checkpoint-") as temp:
        staging = Path(temp) / "archive.zip"
        shutil.copyfile(source, staging)
        with zipfile.ZipFile(staging, "r") as archive:
            verify_zip(archive, with_models=False)
        manifest = {"version": 1, "models": {}, "files": {}}
        with zipfile.ZipFile(staging, "a", compression=zipfile.ZIP_STORED) as archive:
            for model, (dirname, revision, files) in plans.items():
                prefix = f"checkpoints/huggingface/hub/{dirname}"
                ref = f"{prefix}/refs/main"
                encoded = revision.encode("ascii")
                archive.writestr(ref, encoded)
                manifest["files"][ref] = hashlib.sha256(encoded).hexdigest()
                manifest["models"][model] = revision
                for relative, file in sorted(files.items()):
                    name = f"{prefix}/snapshots/{revision}/{relative}"
                    with file.open("rb") as src, archive.open(name, "w", force_zip64=True) as dst:
                        digest = transfer(src, dst)
                    if re.fullmatch(r"[0-9a-f]{64}", file.name) and digest != file.name:
                        raise ValueError(f"Cached blob SHA-256 mismatch: {model}/{relative}")
                    manifest["files"][name] = digest
            archive.writestr(MODEL_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Publish atomically only after closing the central directory. Neither operation
        # replaces an existing destination, including one created by a concurrent run.
        if os.name == "nt":
            os.rename(staging, output)
        else:
            os.link(staging, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Existing benchmark data ZIP")
    parser.add_argument("--cache", type=Path, help="Hugging Face cache root containing hub/")
    parser.add_argument("--output", type=Path, help="New combined ZIP path")
    parser.add_argument("--revision", action="append", default=[], metavar="MODEL=SHA",
                        help="Select the exact cached 40-character commit for a model")
    parser.add_argument("--verify", action="store_true", help="Verify a data ZIP or a combined checkpoint ZIP")
    args = parser.parse_args()
    try:
        if args.verify:
            if args.cache or args.output or args.revision:
                parser.error("--verify takes only the ZIP path")
            with zipfile.ZipFile(args.archive) as archive:
                with_models = verify_zip(archive)
            print("Verified data, results and checkpoint hashes." if with_models else "Verified data and result hashes.")
            return
        if not args.cache or not args.output:
            parser.error("supply --cache HF_HOME and --output NEW_ZIP")
        revisions = {}
        for item in args.revision:
            model, _, revision = item.partition("=")
            if model not in MODELS or not COMMIT.fullmatch(revision) or model in revisions:
                parser.error(f"invalid or repeated model revision: {item}")
            revisions[model] = revision
        extend_archive(args.archive, args.cache.expanduser(), args.output, revisions)
        print(f"Created {args.output}. The recorded revisions are in {MODEL_MANIFEST}.")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        sys.exit(f"Archive error: {exc}")


if __name__ == "__main__":
    main()
