"""日本語Laya APIの速度測定。逐次・SigV4署名・無再試行でrawと集計を保存する。

例: python scripts/benchmark.py --repeat 20 --warmup 1 --out-dir work/benchmark-ja
AWSを呼ばずケースを確認: python scripts/benchmark.py --validate-only
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encode_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def load_cases(path: Path, selected: list[str] | None = None) -> list[dict]:
    fixture = json.loads(path.read_text(encoding="utf-8-sig"))
    if fixture.get("schema_version") != 1:
        raise ValueError("Unsupported fixture schema_version")
    questions = fixture["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 8:
        raise ValueError("Fixture needs 1-8 questions")
    cases = []
    for state in fixture["states"]:
        if not isinstance(state["text"], str) or not state["text"].strip():
            raise ValueError("Each state must contain nonempty text")
        for count in fixture["question_counts"]:
            if type(count) is not int or not 1 <= count <= len(questions):
                raise ValueError("Invalid question count")
            payload = {"state": state["text"], "questions": dict(list(questions.items())[:count])}
            payload_bytes = encode_json(payload).encode("utf-8")
            cases.append({"id": f"{state['id']}-q{count}", "label": state.get("label", state["id"]),
                          "state_chars": len(state["text"]), "question_count": count,
                          "payload_bytes": len(payload_bytes),
                          "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                          "payload": payload})
    ids = [case["id"] for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Cases must be nonempty and have unique ids")
    if selected:
        unknown = set(selected) - set(ids)
        if unknown:
            raise ValueError(f"Unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in selected]
    return cases


def make_schedule(cases: list[dict], warmup: int, repeat: int, seed: int) -> list[tuple]:
    """各ラウンドで各ケースを一度ずつ実行。最初のwarmupは定義順、測定順はseed固定。"""
    scheduled = []
    for iteration in range(1, warmup + 1):
        scheduled.extend(("warmup", iteration, case) for case in cases)
    rng = random.Random(seed)
    for iteration in range(1, repeat + 1):
        shuffled = list(cases)
        rng.shuffle(shuffled)
        scheduled.extend(("measurement", iteration, case) for case in shuffled)
    return scheduled


def percentile(values: list[float], quantile: float) -> float | None:
    """線形補間: sorted[(n-1)*q]。少数標本のp95を高精度推定と解釈しない。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {"n": len(values), "min": min(values), "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95), "max": max(values), "mean": statistics.mean(values)}


def aggregate(records: list[dict]) -> dict:
    return {"requests": len(records), "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "e2e_ms": describe([row["e2e_ms"] for row in records if row.get("e2e_ms") is not None]),
            "prediction_ms": describe([row["prediction_ms"] for row in records
                                       if row.get("prediction_ms") is not None]),
            "outside_prediction_ms": describe([row["outside_prediction_ms"] for row in records
                                               if row.get("outside_prediction_ms") is not None])}


def summarize(records: list[dict], cases: list[dict], run: dict) -> dict:
    groups = {}
    for case in cases:
        all_case = [row for row in records if row["case_id"] == case["id"]]
        measured = [row for row in all_case if row["phase"] == "measurement"]
        success = [row for row in measured if row["success"]]
        groups[case["id"]] = {
            "state_chars": case["state_chars"], "question_count": case["question_count"],
            "all_measurements_including_failures": aggregate(measured),
            "successful_measurements": aggregate(success),
            "successful_measurements_first_flag_false": aggregate([
                row for row in success if row["first_request_in_environment"] is False]),
            "warmup": aggregate([row for row in all_case if row["phase"] == "warmup"]),
        }
    return {"schema_version": 1, "run": run, "ended_at_utc": utc_now(),
            "methodology": {
                "concurrency": 1, "automatic_retries": 0,
                "e2e": "perf_counter: HTTP open through complete response body; excludes credentials, signing, parsing and disk writes",
                "connection": "urllib.request opens a new connection per request; includes DNS/TCP/TLS as applicable, no explicit DNS-cache control",
                "prediction_ms": "Server-reported token-budget validation + agent.predict; excludes model initialization",
                "outside_prediction_ms": "e2e_ms minus prediction_ms; combines network, API, Lambda overhead and any initialization; not a pure network measurement",
                "percentile": "Linear interpolation at (n-1)*q, q=0.50 or 0.95; small-sample p95 is descriptive only",
                "first_flag": "First request in an environment is not proof of a cold start; correlate request_id with CloudWatch INIT/REPORT",
                "order": "Warmup in fixture order, then seeded shuffle of all selected cases each measurement round",
                "accuracy": "Not evaluated; payloads are synthetic Japanese customer-support requests without ground-truth labels",
            },
            "total_requests": len(records), "total_failures": sum(not row["success"] for row in records),
            "first_flag_true_requests": [row["sequence"] for row in records
                                         if row["first_request_in_environment"] is True],
            "by_case": groups}


def call_api(opener, signed_request, timeout: float) -> dict:
    """エラーも一回の試行として返す。署名済みheadersは記録しない。"""
    started = time.perf_counter()
    status, body, headers, transport_error = None, "", {}, None
    try:
        try:
            response = opener.open(signed_request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status = response.status
            headers = dict(response.headers.items())
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        transport_error = {"type": type(exc).__name__, "message": str(getattr(exc, "reason", exc))}
    elapsed = (time.perf_counter() - started) * 1000
    headers_lower = {name.lower(): value for name, value in headers.items()}
    try:
        response_json = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        response_json = None
    response_dict = response_json if isinstance(response_json, dict) else {}
    meta = response_dict.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    prediction_ms = meta.get("prediction_ms")
    if type(prediction_ms) not in (int, float) or not math.isfinite(prediction_ms) or prediction_ms < 0:
        prediction_ms = None
    valid_response = (isinstance(response_dict.get("result"), dict)
                      and bool(response_dict.get("request_id")) and prediction_ms is not None)
    success = status == 200 and valid_response
    return {"status": status, "success": success, "e2e_ms": elapsed,
            "prediction_ms": prediction_ms,
            "outside_prediction_ms": elapsed - prediction_ms if prediction_ms is not None else None,
            "request_id": response_dict.get("request_id"),
            "api_request_id": headers_lower.get("apigw-requestid") or headers_lower.get("x-amzn-requestid"),
            "api_extended_request_id": headers_lower.get("x-amz-apigw-id"),
            "first_request_in_environment": meta.get("first_request_in_environment"),
            "state_tokens": meta.get("state_tokens"), "sequence_tokens": meta.get("sequence_tokens"),
            "model": meta.get("model"), "model_revision": meta.get("model_revision"),
            "laya_version": meta.get("laya_version"), "transport_error": transport_error,
            "response_contract_error": "Expected request_id, result object and finite prediction_ms" if status == 200 and not valid_response else None,
            "response_headers": headers, "response_body": body, "response_json": response_json}


CSV_FIELDS = ["run_id", "sequence", "phase", "iteration", "case_id", "started_at_utc",
              "state_chars", "question_count", "payload_bytes", "payload_sha256", "status", "success",
              "e2e_ms", "prediction_ms", "outside_prediction_ms", "request_id", "api_request_id",
              "api_extended_request_id", "first_request_in_environment", "state_tokens", "sequence_tokens",
              "model", "model_revision", "laya_version", "transport_error", "response_contract_error",
              "request_payload", "response_headers", "response_body"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "cdk-outputs.json")
    parser.add_argument("--stack-name", default="LayaApiStack")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "examples" / "benchmark-ja.json")
    parser.add_argument("--cases", nargs="+", help="e.g. short-q1 medium-q3 long-q8")
    parser.add_argument("--repeat", type=int, default=20, help="Measurement rounds per case (default 20)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds per case, saved but separated (default 1)")
    parser.add_argument("--interval", type=float, default=0.6, help="Seconds after one response before the next request")
    parser.add_argument("--timeout", type=float, default=35, help="HTTP socket timeout in seconds")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--max-consecutive-failures", type=int, default=3, help="Abort after this many failures, 0 disables")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "work" / ("benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--validate-only", action="store_true", help="Print cases and request counts; no AWS access or credentials")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 1000 or not 0 <= args.warmup <= 100:
        parser.error("repeat must be 1-1000 and warmup 0-100")
    if not math.isfinite(args.interval) or args.interval < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("interval must be finite and >= 0; timeout must be finite and > 0")
    if args.max_consecutive_failures < 0:
        parser.error("max-consecutive-failures must be >= 0")
    cases = load_cases(args.fixtures, args.cases)
    schedule = make_schedule(cases, args.warmup, args.repeat, args.seed)
    case_info = [{key: value for key, value in case.items() if key != "payload"} for case in cases]
    if args.validate_only:
        print(json.dumps({"cases": case_info, "planned_requests": len(schedule),
                          "note": "Token budgets must still be checked with the deployed model tokenizer"}, ensure_ascii=False, indent=2))
        return 0

    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    outputs = json.loads(args.outputs.read_text(encoding="utf-8-sig"))[args.stack_name]
    region = args.region or outputs["AwsRegion"]
    base_url = outputs["ApiUrl"]
    from api_endpoint import validate_api_endpoint
    validate_api_endpoint(base_url, region, paths=("", "/"))
    url = base_url.rstrip("/") + "/predict"
    validate_api_endpoint(url, region, paths=("/predict",))
    session = boto3.Session(profile_name=args.profile, region_name=region)
    opener = urllib.request.build_opener(NoRedirect())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("requests.jsonl", "requests.csv", "summary.json", "manifest.json"):
        if (args.out_dir / filename).exists():
            raise ValueError(f"Refusing to overwrite previous run: {args.out_dir / filename}")
    run = {"id": str(uuid.uuid4()), "started_at_utc": utc_now(), "endpoint": url, "region": region,
           "profile": args.profile, "repeat": args.repeat, "warmup": args.warmup, "interval_s": args.interval,
           "socket_timeout_s": args.timeout, "seed": args.seed,
           "planned_requests": len(schedule), "max_consecutive_failures": args.max_consecutive_failures,
           "fixtures_path": str(args.fixtures.resolve()),
           "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "python_version": sys.version, "status": "running"}
    (args.out_dir / "manifest.json").write_text(json.dumps({"run": run, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, consecutive_failures = [], 0
    try:
        with (args.out_dir / "requests.jsonl").open("x", encoding="utf-8") as raw, (args.out_dir / "requests.csv").open("x", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for sequence, (phase, iteration, case) in enumerate(schedule, 1):
                credentials = session.get_credentials()
                if credentials is None:
                    raise RuntimeError("AWS credentials not found; configure an existing profile or sign in with SSO")
                payload = encode_json(case["payload"]).encode("utf-8")
                request = AWSRequest(method="POST", url=url, data=payload,
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
                SigV4Auth(credentials.get_frozen_credentials(), "execute-api", region).add_auth(request)
                signed = urllib.request.Request(url, data=payload, method="POST", headers=dict(request.headers))
                row = {"run_id": run["id"], "sequence": sequence, "phase": phase, "iteration": iteration,
                       "case_id": case["id"], "started_at_utc": utc_now(),
                       **{key: case[key] for key in ("state_chars", "question_count", "payload_bytes", "payload_sha256")},
                       "request_payload": case["payload"], **call_api(opener, signed, args.timeout)}
                rows.append(row)
                raw.write(encode_json(row) + "\n")
                raw.flush()
                writer.writerow({key: encode_json(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
                csv_file.flush()
                print(f"{sequence}/{len(schedule)} {phase} {case['id']} HTTP={row['status']} e2e={row['e2e_ms']:.1f}ms prediction={row['prediction_ms']}ms first={row['first_request_in_environment']}", flush=True)
                consecutive_failures = 0 if row["success"] else consecutive_failures + 1
                if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                    run["status"] = "aborted_consecutive_failures"
                    break
                if sequence < len(schedule):
                    time.sleep(args.interval)
            else:
                run["status"] = "completed"
    except KeyboardInterrupt:
        run["status"] = "interrupted"
    except Exception as exc:
        run["status"] = "runner_error"
        run["runner_error_type"] = type(exc).__name__
        raise
    finally:
        summary = summarize(rows, cases, run)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {len(rows)} requests to {args.out_dir.resolve()}", flush=True)
    return 0 if run["status"] == "completed" and all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
