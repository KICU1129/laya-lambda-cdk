"""Export only explicitly reviewed files. This is a defense, not a secret detector proof.

Local deployment outputs and raw evidence must stay outside PUBLIC_FILES.txt.
No network or AWS access is performed. Findings never include matched content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {".private", ".venv", "venv", "cdk.out", ".git", "work", "outputs",
                 "node_modules", "__pycache__", ".pytest_cache", ".aws"}
PRIVATE_NAMES = {"cdk-outputs.json", "cdk.context.json", "credentials", "config.json",
                 "client-environment.json"}
PRIVATE_JSON_KEYS = {"region", "aws_region", "account", "account_id", "accountid",
    "api_url", "endpoint", "endpoint_url", "model_endpoint", "profile", "aws_profile",
    "function_name", "function_arn", "lambda_arn", "api_id", "request_id", "requestid",
    "log_group", "log_stream", "client_environment", "os", "hostname", "username",
    "started_at_utc", "ended_at_utc", "memory_mb", "lambda_timeout_seconds",
    "lambda_version", "architecture", "provisioned_concurrency_during_run",
    "provisioned_concurrency_ready_elapsed_ms", "cleanup", "first_request_in_environment"}
RULES = {
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "provider-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,})\b"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "account-id": re.compile(r"(?<!\d)\d{12}(?!\d)"),
    "api-host": re.compile(r"\b([a-z0-9-]+)\.execute-api\.[a-z0-9-]+\.amazonaws\.com\b"),
    "aws-arn": re.compile(r"arn:aws(?:-[a-z]+)?:[a-z0-9-]+:[a-z0-9-]*:\d{12}:[^\s\"'<>]+"),
    "windows-private-path": re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|Acro|projects|workspaces)[\\/]"),
    "unix-private-path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)"),
    "secret-assignment": re.compile(
        r'''(?ix)(?:["']?)(?:aws_secret_access_key|aws_session_token|secret_access_key|api_key|access_token|password|client_secret)(?:["']?)\s*[:=]\s*["']([A-Za-z0-9_+/.=\-]{16,})["']'''),
}


def placeholder(value: str) -> bool:
    low = value.lower()
    return (low.startswith(("your-", "your_", "example", "placeholder", "dummy", "replace_"))
            or low in {"api-id", "apiid"})


def safe_display(name: str) -> str:
    # Even an accidentally pasted absolute path in the manifest is not echoed.
    if any(pattern.search(name) for pattern in RULES.values()):
        return "[selected file]"
    return name if re.fullmatch(r"[A-Za-z0-9_.\-/\u3000-\u9fff]+", name) and ":" not in name else "[manifest entry]"


def load_files(root: Path, manifest: Path) -> list[tuple[str, bytes]]:
    root = root.resolve()
    try:
        entries = manifest.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise ValueError("manifest: non-utf8-file") from None
    selected = []
    seen = set()
    for number, raw in enumerate(entries, 1):
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        rel = PurePosixPath(name)
        if ("\\" in name or ":" in name or rel.is_absolute() or ".." in rel.parts
                or not rel.parts or name != rel.as_posix()):
            raise ValueError(f"manifest:{number}: unsafe-path")
        if any(p.lower() in PRIVATE_PARTS or p.lower().startswith(".env") for p in rel.parts):
            raise ValueError(f"manifest:{number}: private-path")
        if rel.name.lower() in PRIVATE_NAMES or rel.suffix.lower() in {".log", ".pem", ".key", ".p12", ".pfx", ".zip"}:
            raise ValueError(f"manifest:{number}: private-file")
        if name.casefold() in seen:
            raise ValueError(f"manifest:{number}: duplicate-path")
        seen.add(name.casefold())
        candidate = root
        for part in rel.parts:
            candidate = candidate / part
            if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
                raise ValueError(f"manifest:{number}: linked-path")
        if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"manifest:{number}: missing-or-non-file")
        selected.append((name, candidate.read_bytes()))
    if not selected:
        raise ValueError("manifest: empty-selection")
    return selected


def scan_file(name: str, payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [f"{safe_display(name)}:1: non-utf8-file"]
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in RULES.items():
            for match in pattern.finditer(line):
                candidate = match.group(1) if rule in {"secret-assignment", "api-host"} else match.group()
                if rule == "account-id" and len(set(candidate)) == 1:
                    continue
                if rule in {"account-id", "api-host", "secret-assignment"} and placeholder(candidate):
                    continue
                if rule == "aws-arn":
                    account = match.group().split(":")[4]
                    if len(set(account)) == 1:
                        continue
                findings.append(f"{safe_display(name)}:{number}: {rule}")
                break
    # Runtime evidence has a stronger policy than generic source code examples.
    measured = name.startswith("docs/") and "/data/" in name
    if measured and name.endswith((".json", ".jsonl")):
        try:
            objects = [json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith(".jsonl") else [json.loads(text)]
        except json.JSONDecodeError:
            findings.append(f"{safe_display(name)}:1: invalid-json")
            return findings
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower() in PRIVATE_JSON_KEYS:
                        pattern = re.compile(r'"' + re.escape(key) + r'"\s*:')
                        number = next((i for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)), 1)
                        findings.append(f"{safe_display(name)}:{number}: private-evidence-field")
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        for obj in objects:
            walk(obj)
    return sorted(set(findings))


def export(root: Path, manifest: Path, out_dir: Path | None = None,
           zip_path: Path | None = None, check_only: bool = False) -> int:
    files = load_files(root, manifest)
    findings = [finding for name, payload in files for finding in scan_file(name, payload)]
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    if check_only:
        print(f"PASS: {len(files)} explicitly selected text files checked; no package written.")
        return 0
    if out_dir is None and zip_path is None:
        raise ValueError("destination-required: specify --out-dir or --zip")
    project = root.resolve()
    for target in [out_dir, zip_path]:
        if target is None:
            continue
        resolved = target.resolve()
        if resolved == project or resolved.is_relative_to(project):
            raise ValueError("unsafe-destination: use a destination outside the project")
        for ancestor in [target, *target.parents]:
            if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
                raise ValueError("unsafe-destination: linked-path")
    if out_dir is not None and out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError("destination-not-empty")
    if zip_path is not None and zip_path.exists():
        raise ValueError("zip-already-exists")
    if out_dir is not None and zip_path is not None and zip_path.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError("zip-inside-output-directory")
    checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files).encode("utf-8")
    if any(name.casefold() == "sha256sums.txt" for name, _ in files):
        raise ValueError("reserved-output-name")
    files.append(("SHA256SUMS.txt", checksums))
    # Read and inspect once; package exactly those bytes, not files changed after the check.
    with tempfile.TemporaryDirectory(prefix="public-export-") as temp:
        staging = Path(temp) / "package"
        staging.mkdir()
        for name, data in files:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging, out_dir, dirs_exist_ok=True)
        if zip_path is not None:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in files:
                    archive.writestr(name, data)
    print(f"PASS: exported {len(files) - 1} selected files plus SHA256SUMS.txt.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "PUBLIC_FILES.txt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        return export(ROOT, args.manifest, args.out_dir, args.zip_path, args.check_only)
    except ValueError as exc:
        # All ValueError messages above are authored safe diagnostics.
        print(str(exc), file=sys.stderr)
        return 2
    except OSError:
        print("filesystem-error: check local paths and permissions", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
