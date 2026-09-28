import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("public_export", Path(__file__).resolve().parents[1] / "scripts/export_public.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture(tmp_path, entries=None):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# Public\n日本語の評価結果\n", encoding="utf-8")
    manifest = root / "PUBLIC_FILES.txt"
    manifest.write_text("\n".join(entries or ["README.md"]), encoding="utf-8")
    return root, manifest


def test_export_is_exact_allowlist_and_preserves_review_values(tmp_path):
    rel = "docs/benchmarks/reviews/data/results.jsonl"
    root, manifest = fixture(tmp_path, ["README.md", rel])
    secret = root / "cdk-outputs.json"
    secret.write_text("private deployment evidence", encoding="utf-8")
    result = root / rel
    result.parent.mkdir(parents=True)
    row = {"id": "R001", "text": "使いやすいです。", "expected": "positive", "predicted": "positive", "correct": True, "client_ms": 123.45}
    result.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    destination = tmp_path / "publish"
    archive = tmp_path / "public.zip"
    assert exporter.export(root, manifest, destination, archive) == 0
    assert json.loads((destination / rel).read_text(encoding="utf-8")) == row
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"README.md", rel, "SHA256SUMS.txt"}
    assert not (destination / "cdk-outputs.json").exists()


@pytest.mark.parametrize("entry", ["../outside.md", "/absolute.md", "cdk.out/file.json", ".private/file.json", ".env", "cdk-outputs.json", "README.md\nREADME.md"])
def test_unsafe_selection_rejected(tmp_path, entry):
    root, manifest = fixture(tmp_path, [entry])
    with pytest.raises(ValueError):
        exporter.export(root, manifest, check_only=True)


def test_sensitive_matches_never_echo_values_or_make_package(tmp_path, capsys):
    root, manifest = fixture(tmp_path)
    sensitive = "AK" + "IA" + "B" * 16
    (root / "README.md").write_text(sensitive, encoding="utf-8")
    destination = tmp_path / "publish"
    assert exporter.export(root, manifest, destination) == 1
    assert not destination.exists()
    diagnostic = capsys.readouterr().err
    assert sensitive not in diagnostic
    assert "README.md:1: aws-access-key" in diagnostic


@pytest.mark.parametrize("key", ["region", "profile", "started_at_utc", "client_environment", "memory_mb"])
def test_private_evidence_fields_fail_even_if_values_are_unrecognized(key):
    text = json.dumps({"nested": [{key: "arbitrary"}]})
    assert exporter.scan_file("docs/benchmarks/reviews/data/manifest.json", text.encode())


def test_dummy_fixture_and_generic_docs_are_accepted():
    text = 'account="111111111111"\nendpoint="https://api-id.execute-api.region.amazonaws.com"\napi_key="your_secret_placeholder"\n'
    assert exporter.scan_file("tests/example.py", text.encode()) == []


def test_sanitized_warmup_export_allowed_but_raw_metadata_blocked(tmp_path):
    name = "docs/benchmarks/reviews/data/warmup.json"
    root, manifest = fixture(tmp_path, [name])
    target = root / name
    target.parent.mkdir(parents=True)
    clean = {"model": "example/model", "predicted": "positive", "client_ms": 125.5, "success": True}
    target.write_text(json.dumps(clean), encoding="utf-8")
    assert exporter.export(root, manifest, check_only=True) == 0
    raw = {**clean, "started_at_utc": "private", "first_request_in_environment": True}
    target.write_text(json.dumps(raw), encoding="utf-8")
    assert exporter.export(root, manifest, check_only=True) == 1


@pytest.mark.parametrize("value,rule", [
    ("123456" + "789013", "account-id"),
    ("https://" + "uniquehost" + ".execute-api." + "us-east-1.amazonaws.com", "api-host"),
    ("arn:aws:lambda:us-east-1:" + "123456" + "789013:function:example", "aws-arn"),
    ("C:" + "/Users/" + "testuser/work", "windows-private-path"),
    ("/" + "home/testuser/work", "unix-private-path"),
    ('api_key="' + "b" * 30 + '"', "secret-assignment"),
])
def test_sensitive_content_rules_redact_injected_values(value, rule):
    findings = exporter.scan_file("README.md", value.encode())
    assert any(item.endswith(": " + rule) for item in findings)
    assert all(value not in item for item in findings)


def test_destinations_cannot_overwrite_or_modify_input(tmp_path):
    root, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="unsafe-destination"):
        exporter.export(root, manifest, root / "publish")
    destination = tmp_path / "publish"
    destination.mkdir()
    keep = destination / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="destination-not-empty"):
        exporter.export(root, manifest, destination)
    assert keep.read_text() == "preserve"
    archive = tmp_path / "public.zip"
    archive.write_bytes(b"existing")
    with pytest.raises(ValueError, match="zip-already-exists"):
        exporter.export(root, manifest, zip_path=archive)
    assert archive.read_bytes() == b"existing"


def test_symlink_file_is_rejected(tmp_path):
    root, manifest = fixture(tmp_path, ["linked.md"])
    try:
        (root / "linked.md").symlink_to(root / "README.md")
    except OSError:
        pytest.skip("OS does not permit unprivileged symlinks")
    with pytest.raises(ValueError, match="linked-path"):
        exporter.export(root, manifest, check_only=True)
