"""正解を事前に付けた日本語問い合わせを、IAM認証付きの実APIで評価する。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("department", "urgency", "refund_requested", "duplicate_charge",
        "cancellation_requested", "service_unavailable", "contact_channel", "sentiment")
BOOL_KEYS = ("refund_requested", "duplicate_charge", "cancellation_requested", "service_unavailable")
SCORE_KEYS = ("urgency", "sentiment")
CHOICE_OPTIONS = {"department": {"billing", "technical", "other"},
                  "contact_channel": {"email", "phone", "unspecified"}}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_fixture(path: Path):
    fixture = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = fixture.get("cases")
    if fixture.get("schema_version") != 1 or not isinstance(cases, list) or not cases:
        raise ValueError("Expected schema_version=1 and nonempty cases")
    ids = set()
    for case in cases:
        case_id, state, gold = case["id"], case["state"], case["gold"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Missing or duplicate case id")
        ids.add(case_id)
        if not isinstance(state, str) or not state.strip() or len(state) > 12000:
            raise ValueError(f"Invalid state: {case_id}")
        if set(gold) != set(KEYS):
            raise ValueError(f"Expected exactly eight gold labels: {case_id}")
        for key, options in CHOICE_OPTIONS.items():
            if gold[key] not in options:
                raise ValueError(f"Invalid {key}: {case_id}")
        for key in BOOL_KEYS:
            if type(gold[key]) is not bool:
                raise ValueError(f"Invalid boolean {key}: {case_id}")
        for key in SCORE_KEYS:
            if type(gold[key]) is not int or gold[key] not in (0, 1, 2):
                raise ValueError(f"Invalid score {key}: {case_id}")
    return fixture


def wilson_interval(correct: int, total: int):
    if total == 0:
        return None
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    middle = (p + z*z/(2*total)) / denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [round(middle-half, 4), round(middle+half, 4)]


def predicted(answer: dict, key: str):
    if not isinstance(answer, dict):
        raise ValueError(f"Missing answer {key}")
    if key in BOOL_KEYS:
        p = answer.get("noul")
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid noul probability {key}")
        label = p >= 0.5
        return label, (p if label else 1-p)
    if key in SCORE_KEYS:
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != {"0", "1", "2"}:
            raise ValueError(f"Invalid score probabilities {key}")
        best = max(probabilities, key=probabilities.__getitem__)
        return int(best), probabilities[best]
    label = answer.get("choice")
    probabilities = answer.get("probabilities")
    if label not in CHOICE_OPTIONS[key] or not isinstance(probabilities, dict) or label not in probabilities:
        raise ValueError(f"Invalid choice {key}")
    return label, probabilities[label]


def score(records: list[dict], fixture: dict):
    rows = []
    by_id = {case["id"]: case for case in fixture["cases"]}
    for record in records:
        case = by_id[record["id"]]
        if not record["success"]:
            continue
        answer_set = record["response"]["result"]["answers"]
        for key in KEYS:
            value, confidence = predicted(answer_set.get(key), key)
            rows.append({"id": case["id"], "question": key, "state": case["state"],
                         "rationale": case.get("rationale"), "gold": case["gold"][key],
                         "predicted": value, "correct": value == case["gold"][key],
                         "predicted_probability": confidence,
                         "yes_probability": answer_set[key].get("noul") if key in BOOL_KEYS else None,
                         "score_value": answer_set[key].get("score") if key in SCORE_KEYS else None})
    metrics = {}
    for key in KEYS:
        selected = [row for row in rows if row["question"] == key]
        n, correct = len(selected), sum(row["correct"] for row in selected)
        values = {"n": n, "correct": correct,
                  "accuracy": round(correct/n, 4) if n else None,
                  "wilson_95": wilson_interval(correct, n),
                  "gold_distribution": {}, "predicted_distribution": {}, "confusion": {}}
        for row in selected:
            gold, prediction = str(row["gold"]), str(row["predicted"])
            values["gold_distribution"][gold] = values["gold_distribution"].get(gold, 0) + 1
            values["predicted_distribution"][prediction] = values["predicted_distribution"].get(prediction, 0) + 1
            values["confusion"].setdefault(gold, {})[prediction] = values["confusion"].setdefault(gold, {}).get(prediction, 0) + 1
        if key in BOOL_KEYS:
            tp = sum(row["gold"] is True and row["predicted"] is True for row in selected)
            fp = sum(row["gold"] is False and row["predicted"] is True for row in selected)
            fn = sum(row["gold"] is True and row["predicted"] is False for row in selected)
            tn = sum(row["gold"] is False and row["predicted"] is False for row in selected)
            values.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": round(tp/(tp+fp), 4) if tp+fp else None,
                           "recall": round(tp/(tp+fn), 4) if tp+fn else None,
                           "f1": round(2*tp/(2*tp+fp+fn), 4) if 2*tp+fp+fn else None})
        if key in SCORE_KEYS and n:
            values["score_mae"] = round(sum(abs(float(row["score_value"])-row["gold"])
                                              for row in selected) / n, 4)
        metrics[key] = values
    total = len(rows)
    complete = [case["id"] for case in fixture["cases"] if sum(
        row["correct"] for row in rows if row["id"] == case["id"]) == len(KEYS)]
    return rows, {"requests": len(records), "successful_requests": sum(r["success"] for r in records),
                  "failed_requests": sum(not r["success"] for r in records),
                  "decisions": total, "correct_decisions": sum(row["correct"] for row in rows),
                  "micro_accuracy": round(sum(row["correct"] for row in rows)/total, 4) if total else None,
                  "exact_match_all_8": {"correct": len(complete), "total": len(fixture["cases"]),
                                        "rate": round(len(complete)/len(fixture["cases"]), 4)},
                  "by_question": metrics,
                  "errors": sorted([row for row in rows if not row["correct"]],
                                   key=lambda row: row["predicted_probability"], reverse=True)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call(opener, signed, timeout):
    started = time.perf_counter()
    try:
        with opener.open(signed, timeout=timeout) as response:
            status, body = response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        return {"status": None, "response": None, "error": str(exc),
                "elapsed_ms": round((time.perf_counter()-started)*1000, 2), "success": False}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    ok = (status == 200 and isinstance(parsed, dict)
          and isinstance(parsed.get("result"), dict)
          and isinstance(parsed["result"].get("answers"), dict)
          and set(parsed["result"]["answers"]) == set(KEYS))
    return {"status": status, "response": parsed if isinstance(parsed, dict) else body,
            "elapsed_ms": round((time.perf_counter()-started)*1000, 2), "success": ok}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "examples/accuracy-ja.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "examples/benchmark-ja.json")
    parser.add_argument("--outputs", type=Path, default=ROOT / "cdk-outputs.json")
    parser.add_argument("--out-dir", type=Path, required=False)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--interval", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=35)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("invalid interval or timeout")
    fixture = load_fixture(args.fixture)
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))["questions"]
    if list(questions) != list(KEYS):
        raise ValueError("Question set drift; revalidate gold labels")
    if args.validate_only:
        print(encode({"cases": len(fixture["cases"]), "ids": [x["id"] for x in fixture["cases"]],
                      "gold_distributions": {key: {str(value): sum(case["gold"][key] == value for case in fixture["cases"])
                                                   for value in {case["gold"][key] for case in fixture["cases"]}}
                                             for key in KEYS}}))
        return 0
    if args.out_dir is None:
        parser.error("--out-dir is required for AWS calls")
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    outputs = json.loads(args.outputs.read_text(encoding="utf-8-sig"))["LayaApiStack"]
    url, region = outputs["PredictUrl"], outputs["AwsRegion"]
    from api_endpoint import validate_api_endpoint
    validate_api_endpoint(url, region, paths=("/predict",))
    session = boto3.Session(profile_name=args.profile, region_name=region)
    opener = urllib.request.build_opener(NoRedirect())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "requests.jsonl"
    if any((args.out_dir / name).exists() for name in ("requests.jsonl", "decisions.csv", "summary.json")):
        raise ValueError("Refusing to overwrite a prior evaluation")
    manifest = {"started_at_utc": now_utc(), "model_endpoint": url, "region": region,
                "dataset_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                "question_keys": list(KEYS), "cases": len(fixture["cases"]),
                "sequential": True, "automatic_retries": 0, "interval_s": args.interval,
                "noul_threshold": 0.5, "score_decision": "argmax(probabilities)",
                "limits": "Synthetic Japanese support tickets with labels authored before inference; no independent human annotation or production sampling."}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    try:
        with raw_path.open("x", encoding="utf-8") as handle:
            for index, case in enumerate(fixture["cases"], 1):
                credentials = session.get_credentials()
                if credentials is None:
                    raise RuntimeError("AWS credentials not found")
                payload = encode({"state": case["state"], "questions": questions}).encode("utf-8")
                request = AWSRequest(method="POST", url=url, data=payload,
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
                SigV4Auth(credentials.get_frozen_credentials(), "execute-api", region).add_auth(request)
                signed = urllib.request.Request(url, data=payload, method="POST", headers=dict(request.headers))
                result = {"id": case["id"], "started_at_utc": now_utc(),
                          "state": case["state"], "gold": case["gold"], **call(opener, signed, args.timeout)}
                records.append(result)
                handle.write(encode(result) + "\n")
                handle.flush()
                print(f"{index}/{len(fixture['cases'])} {case['id']} HTTP {result['status']} {result['elapsed_ms']:.1f}ms", flush=True)
                if index < len(fixture["cases"]):
                    time.sleep(args.interval)
    finally:
        decisions, report = score(records, fixture)
        report["manifest"] = manifest
        report["ended_at_utc"] = now_utc()
        (args.out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (args.out_dir / "decisions.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            fields = ("id", "question", "gold", "predicted", "correct", "predicted_probability",
                      "yes_probability", "score_value", "state", "rationale")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(decisions)
    return 0 if all(record["success"] for record in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
