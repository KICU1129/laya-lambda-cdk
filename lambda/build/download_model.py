"""公開モデルを固定commitから取得し、Lambdaで書き込み不要にする。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def normalize_tokenizer_config(path: Path) -> bool:
    """Laya起動時に実施されるtokenizer互換補正をビルド時に済ませる。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if config.get("tokenizer_class") in (None, "TokenizersBackend"):
        config["tokenizer_class"] = "PreTrainedTokenizerFast"
        config.pop("backend", None)
        config.pop("is_local", None)
        changed = True
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        config["extra_special_tokens"] = {f"extra_{i}": value for i, value in enumerate(extra)}
        changed = True
    if changed:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.lock.read_text(encoding="utf-8"))
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model.lock.json revision must be a full 40-character commit SHA")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"], revision=revision, local_dir=str(root),
        allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*.json",
                        "tokenizer/*", "README.md", "LICENSE*", "NOTICE*"],
        max_workers=4,
    )
    required = ["rl_agent_config.json", "model.safetensors", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required model artifact missing: {name}")
    # local_dirのダウンロードメタデータは実行時に不要。
    shutil.rmtree(root / ".cache", ignore_errors=True)
    changed = normalize_tokenizer_config(root / "tokenizer/tokenizer_config.json")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                hashes[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "repo_id": spec["repo_id"], "revision": revision,
        "tokenizer_config_normalized": changed,
        "sha256": hashes,
    }
    (root / "build-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 読み取り専用でロードできることを後続の非root smoke testで確認する。
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps({"downloaded_model": spec["repo_id"], "revision": revision,
                      "files": len(hashes), "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
