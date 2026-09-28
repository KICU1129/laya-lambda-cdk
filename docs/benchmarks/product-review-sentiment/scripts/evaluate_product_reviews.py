"""Run a fixed Japanese sentiment benchmark; retain only publishable response fields."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import urllib.error
import urllib.request

LABELS = ("positive", "neutral", "negative")
NAMES = dict(zip(LABELS, ("肯定", "中立", "否定")))
QUESTION = {"sentiment": {
    "type": "choice",
    "instructions": "この商品レビューに表れている、商品に対する全体的な感情を選んでください。",
    "criteria": {"positive": "肯定的。満足、評価、好意が全体として優勢。",
                 "neutral": "中立。事実の記述が中心、または良い点と悪い点が同程度で評価が定まらない。",
                 "negative": "否定的。不満、失望、批判が全体として優勢。"}}}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(data):
    cases = data["cases"]
    if len(cases) != 100 or len({c["id"] for c in cases}) != 100:
        raise ValueError("Exactly 100 unique cases required")
    if len({c["text"] for c in cases}) != 100:
        raise ValueError("Duplicate review")
    for case in cases:
        if case["expected"] not in LABELS or not case["text"].strip() or not case["rationale"].strip():
            raise ValueError("Invalid case")
    return cases


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    x = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def timings(values):
    return {"n": len(values), **{name: round(value, 2) if value is not None else None
        for name, value in {"min": min(values) if values else None,
                            "mean": statistics.mean(values) if values else None,
                            "p50": percentile(values, 50), "p95": percentile(values, 95),
                            "max": max(values) if values else None}.items()}}


def summarize(cases, records):
    expected = {c["id"]: c["expected"] for c in cases}
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or any(i not in expected for i in ids):
        raise ValueError("Unexpected or repeated result ID")
    matrix = {g: {p: 0 for p in (*LABELS, "error")} for g in LABELS}
    by_id = {r["id"]: r for r in records}
    for case in cases:
        r = by_id.get(case["id"])
        matrix[case["expected"]][r["predicted"] if r and r["success"] else "error"] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[g][label] for g in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"support": support, "correct": tp, "predicted": predicted,
                            "precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    successful = [r for r in records if r["success"]]
    correct = sum(matrix[g][g] for g in LABELS)
    return {"cases": len(cases), "attempted": len(records), "successful": len(successful),
            "failed_or_missing": len(cases) - len(successful), "correct": correct,
            "accuracy": correct / len(cases),
            "successful_only_accuracy": correct / len(successful) if successful else None,
            "macro_f1": statistics.mean(v["f1"] for v in per_class.values()),
            "majority_baseline": max(Counter(expected.values()).values()) / len(cases),
            "confusion": matrix, "per_class": per_class,
            "client_latency_ms": timings([r["client_ms"] for r in successful]),
            "prediction_latency_ms": timings([r["prediction_ms"] for r in successful if r["prediction_ms"] is not None]),
            "all_attempt_latency_ms": timings([r["client_ms"] for r in records])}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def infer(session, url, region, text, timeout=35):
    from api_endpoint import validate_api_endpoint
    validate_api_endpoint(url, region, paths=("/predict",))
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    payload = json.dumps({"state": text, "questions": QUESTION}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials unavailable")
    request = AWSRequest(method="POST", url=url, data=payload, headers={"Content-Type": "application/json"})
    SigV4Auth(credentials.get_frozen_credentials(), "execute-api", region).add_auth(request)
    signed = urllib.request.Request(url, data=payload, headers=dict(request.headers), method="POST")
    opener = urllib.request.build_opener(NoRedirect())
    status, response, error = None, {}, None
    started_at = utc()
    started = time.perf_counter()
    try:
        with opener.open(signed, timeout=timeout) as result:
            status = result.status
            raw = result.read()
        elapsed = (time.perf_counter() - started) * 1000
        response = json.loads(raw)
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        status, error = exc.code, "http_error"
    except (OSError, ValueError) as exc:
        elapsed = (time.perf_counter() - started) * 1000
        error = type(exc).__name__
    answer = response.get("result", {}).get("answers", {}).get("sentiment", {})
    meta = response.get("meta", {})
    probabilities = answer.get("probabilities", {})
    choice = answer.get("choice")
    valid = (status == 200 and choice in LABELS and set(probabilities) == set(LABELS)
             and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
             and abs(sum(probabilities.values()) - 1) < 0.002
             and probabilities[choice] >= max(probabilities.values()) - 0.00011)
    # Whitelist only: never persist URL, AWS headers, request IDs or error bodies.
    return {"started_at_utc": started_at, "http_status": status, "success": valid,
            "error": error if error else (None if valid else "invalid_answer"),
            "predicted": choice if valid else None, "probabilities": probabilities if valid else {},
            "client_ms": round(elapsed, 2), "prediction_ms": meta.get("prediction_ms"),
            "first_request_in_environment": meta.get("first_request_in_environment"),
            "model": meta.get("model"), "model_revision": meta.get("model_revision"),
            "laya_version": meta.get("laya_version"), "state_tokens": meta.get("state_tokens"),
            "sequence_tokens": meta.get("sequence_tokens"),
            "usage": response.get("result", {}).get("usage")}


def run(args):
    import boto3
    from botocore.exceptions import ClientError
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    cases = validate(data)
    settings = json.loads(args.outputs.read_text(encoding="utf-8-sig"))["LayaApiStack"]
    from api_endpoint import validate_api_endpoint
    validate_api_endpoint(settings["PredictUrl"], settings["AwsRegion"], paths=("/predict",))
    session = boto3.Session(profile_name=args.profile, region_name=settings["AwsRegion"])
    client = session.client("lambda")
    target = {"FunctionName": settings["FunctionName"], "Qualifier": "live"}
    try:
        client.get_provisioned_concurrency_config(**target)
        raise RuntimeError("Existing provisioned concurrency found; refusing to replace it")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ProvisionedConcurrencyConfigNotFoundException":
            raise
    config = client.get_function_configuration(**target)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dump(args.out_dir / "client-environment.json", {"python": platform.python_version(), "boto3": boto3.__version__, "os": platform.system()})
    dump(args.out_dir / "dataset.json", data)
    dump(args.out_dir / "question.json", QUESTION)
    order = cases.copy()
    random.Random(20260928).shuffle(order)
    manifest = {"started_at_utc": utc(), "dataset_sha256": digest(args.out_dir / "dataset.json"),
                "question_sha256": digest(args.out_dir / "question.json"),
                "region": settings["AwsRegion"], "memory_mb": config["MemorySize"],
                "lambda_timeout_seconds": config["Timeout"], "lambda_version": config["Version"],
                "architecture": config["Architectures"], "provisioned_concurrency_during_run": 1,
                "request_order": [c["id"] for c in order], "shuffle_seed": 20260928,
                "repeats_per_case": 1, "automatic_inference_retries": 0, "concurrency": 1,
                "interval_seconds": 0.6, "client_timeout_seconds": 35,
                "warmup_requests": 1, "warmup_text": "卓上の小さな収納箱です。仕切りは取り外せます。",
                "label_source": "AI-authored synthetic cases; labels fixed before inference; independently reviewed by a second AI; no human annotation",
                "latency_definition": "Client: immediately before urllib open through complete response bytes; excludes SigV4 signing, parsing and sleep. Server: meta.prediction_ms.",
                "decision_rule": "API choice, validated against maximum returned probability; no tuning or post-hoc relabeling"}
    records = []
    dump(args.out_dir / "manifest.json", manifest)
    pc_created = False
    try:
        start = time.perf_counter()
        # Cleanup also runs if the create response is interrupted after AWS accepts it.
        pc_created = True
        client.put_provisioned_concurrency_config(**target, ProvisionedConcurrentExecutions=1)
        while True:
            status = client.get_provisioned_concurrency_config(**target)
            print("Initialization:", status["Status"], flush=True)
            if status["Status"] == "READY":
                break
            if status["Status"] == "FAILED" or time.perf_counter() - start > 600:
                raise RuntimeError("Provisioned concurrency initialization failed or exceeded 600s")
            time.sleep(10)
        manifest["provisioned_concurrency_ready_elapsed_ms"] = round((time.perf_counter() - start)*1000, 2)
        warmup = infer(session, settings["PredictUrl"], settings["AwsRegion"], manifest["warmup_text"])
        dump(args.out_dir / "warmup.json", warmup)
        if not warmup["success"]:
            raise RuntimeError("Warmup failed; no scored requests attempted")
        with (args.out_dir / "results.jsonl").open("x", encoding="utf-8") as handle:
            for i, case in enumerate(order, 1):
                result = {"id": case["id"], "expected": case["expected"],
                          **infer(session, settings["PredictUrl"], settings["AwsRegion"], case["text"])}
                result["correct"] = result["success"] and result["predicted"] == case["expected"]
                records.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{i}/100 {case['id']} {result['predicted']} correct={result['correct']} {result['client_ms']}ms", flush=True)
                if i < len(order):
                    time.sleep(0.6)
    finally:
        manifest["ended_at_utc"] = utc()
        if pc_created:
            try:
                client.delete_provisioned_concurrency_config(**target)
                try:
                    client.get_provisioned_concurrency_config(**target)
                    manifest["cleanup"] = "delete_requested_but_configuration_still_present"
                except ClientError as exc:
                    if exc.response["Error"]["Code"] != "ProvisionedConcurrencyConfigNotFoundException":
                        raise
                    manifest["cleanup"] = "deleted_and_absence_verified"
            except Exception:
                manifest["cleanup"] = "cleanup_failed_requires_attention"
                raise
            finally:
                dump(args.out_dir / "manifest.json", manifest)
                dump(args.out_dir / "summary.json", summarize(cases, records))
        else:
            dump(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summarize(cases, records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True, help="Private CDK outputs; never copied to report")
    parser.add_argument("--out-dir", type=Path, required=True, help="New result directory")
    parser.add_argument("--profile")
    run(parser.parse_args())
