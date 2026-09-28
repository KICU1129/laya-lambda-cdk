"""Project private sentiment measurements onto the public data schema."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_product_reviews import summarize, validate

MANIFEST_FIELDS = (
    "dataset_sha256", "question_sha256", "request_order", "shuffle_seed",
    "repeats_per_case", "automatic_inference_retries", "concurrency", "interval_seconds",
    "client_timeout_seconds", "warmup_requests", "warmup_text", "label_source",
    "latency_definition", "decision_rule",
)
RESULT_FIELDS = (
    "id", "expected", "http_status", "success", "error", "predicted", "probabilities",
    "client_ms", "prediction_ms", "model", "model_revision", "laya_version",
    "state_tokens", "sequence_tokens", "usage", "correct",
)


def project_record(record):
    # Fixed fields only: newly introduced private metadata never passes implicitly.
    result = {k: record[k] for k in RESULT_FIELDS if k in record}
    for key, allowed in {"probabilities": ("positive", "neutral", "negative"),
                         "sequence_tokens": ("sentiment",),
                         "usage": ("input_tokens", "output_tokens")}.items():
        if key in result and isinstance(result[key], dict):
            result[key] = {k: result[key][k] for k in allowed if k in result[key]}
    # Error messages may contain URLs or local paths; retain only a generic state.
    result["error"] = None if record.get("success") else "request_failed"
    return result


def sanitize(source: Path, target: Path):
    def read(name):
        return json.loads((source / name).read_text(encoding="utf-8"))
    data, manifest, question = read("dataset.json"), read("manifest.json"), read("question.json")
    cases = validate(data)
    for name, key in (("dataset.json", "dataset_sha256"), ("question.json", "question_sha256")):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Frozen data hash mismatch")
    records = [project_record(json.loads(line)) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary = summarize(cases, records)
    if summary != read("summary.json"):
        raise ValueError("Summary changed during projection")
    target.mkdir(parents=True, exist_ok=False)
    for name in ("dataset.json", "question.json"):
        (target / name).write_bytes((source / name).read_bytes())
    public_manifest = {key: manifest[key] for key in MANIFEST_FIELDS}
    public_manifest["publication_note"] = "Deployment identifiers, timestamps and machine configuration removed. Predictions and latency values unchanged."
    def write(name, value):
        (target / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write("manifest.json", public_manifest)
    write("summary.json", summary)
    write("warmup.json", project_record(read("warmup.json")))
    (target / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print("Public measurement fields exported; review the dataset text before publication.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sanitize(args.input, args.out_dir)
