"""LayaベンチマークをCloudWatchのLambda/APIログと照合する(read-only)。

python work/collect_evidence.py --outputs <cdk-outputs.json> --run-dir <first> \
    --run-dir <benchmark> --out-dir <evidence>
python work/collect_evidence.py --self-test  # AWS/認証情報へのアクセスなし
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import sys
import unittest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def timestamp_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return int(parsed.timestamp() * 1000)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def decode_log_json(message: str) -> list[dict]:
    """通常のJSON、Python logger prefix、Lambda JSONログのmessage入れ子を読む。"""
    found = []
    start = message.find("{")
    if start < 0:
        return found
    try:
        value, _ = json.JSONDecoder().raw_decode(message[start:])
    except (ValueError, TypeError):
        return found
    if isinstance(value, dict):
        found.append(value)
        nested = value.get("message")
        if isinstance(nested, str):
            found.extend(decode_log_json(nested))
        elif isinstance(nested, dict):
            found.append(nested)
    return found


def text_number(message: str, label: str):
    match = re.search(r"(?<!\w)" + re.escape(label) + r":\s*([\d.]+)", message)
    return number(match.group(1)) if match else None


def parse_report(message: str) -> dict | None:
    for data in decode_log_json(message):
        if data.get("type") == "platform.report":
            record = data.get("record", {})
            metrics = record.get("metrics", {})
            return {"request_id": record.get("requestId"),
                    "duration_ms": number(metrics.get("durationMs")),
                    "billed_duration_ms": number(metrics.get("billedDurationMs")),
                    "memory_mb": number(metrics.get("memorySizeMB")),
                    "max_memory_mb": number(metrics.get("maxMemoryUsedMB")),
                    "init_duration_ms": number(metrics.get("initDurationMs")),
                    "status": record.get("status"), "error_type": record.get("errorType"),
                    "format": "platform.report"}
    request_id = re.search(r"\bREPORT\s+RequestId:\s*(\S+)", message)
    if not request_id:
        return None
    status = re.search(r"\bStatus:\s*(\S+)", message)
    error = re.search(r"\bError Type:\s*(\S+)", message)
    return {"request_id": request_id.group(1),
            "duration_ms": text_number(message, "Duration"),
            "billed_duration_ms": text_number(message, "Billed Duration"),
            "memory_mb": text_number(message, "Memory Size"),
            "max_memory_mb": text_number(message, "Max Memory Used"),
            "init_duration_ms": text_number(message, "Init Duration"),
            "status": status.group(1) if status else None,
            "error_type": error.group(1) if error else None, "format": "text"}


def event_metadata(event: dict) -> dict:
    return {key: event.get(key) for key in ("timestamp", "ingestionTime", "logStreamName", "eventId")}


def parse_lambda_events(events: list[dict]) -> dict:
    reports, model_loaded, init_reports, completed = [], [], [], []
    for event in events:
        message = event.get("message", "")
        source = event_metadata(event)
        report = parse_report(message)
        if report:
            reports.append({**report, **source})
        if re.search(r"\bINIT_REPORT\b", message):
            status = re.search(r"\bStatus:\s*(\S+)", message)
            phase = re.search(r"\bPhase:\s*(\S+)", message)
            init_reports.append({**source, "init_duration_ms": text_number(message, "Init Duration"),
                                 "status": status.group(1) if status else None,
                                 "phase": phase.group(1) if phase else None, "message": message})
        for data in decode_log_json(message):
            if data.get("event") == "model_loaded":
                model_loaded.append({**data, **source})
            elif data.get("event") == "request_completed":
                completed.append({**data, **source})
            elif data.get("type") in ("platform.initReport", "platform.initRuntimeDone", "platform.initStart"):
                init_reports.append({**source, "platform_event": data})
    return {"reports": reports, "model_loaded": model_loaded,
            "init_events": init_reports, "request_completed": completed}


def parse_api_events(events: list[dict]) -> list[dict]:
    parsed = []
    for event in events:
        for data in decode_log_json(event.get("message", "")):
            if data.get("requestId"):
                parsed.append({**data, **event_metadata(event),
                               "integration_latency_ms": number(data.get("integrationLatency"))})
    return parsed


def index_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    result = {}
    for record in records:
        value = record.get(key)
        if value:
            result.setdefault(value, []).append(record)
    return result


def join_evidence(raw: list[dict], parsed_lambda: dict, parsed_api: list[dict]) -> tuple[list[dict], dict]:
    report_index = index_by(parsed_lambda["reports"], "request_id")
    completed_index = index_by(parsed_lambda["request_completed"], "request_id")
    api_index = index_by(parsed_api, "requestId")
    model_index = index_by(parsed_lambda["model_loaded"], "logStreamName")
    joined = []
    for request in raw:
        lambda_matches = report_index.get(request.get("request_id"), [])
        api_matches = api_index.get(request.get("api_request_id"), [])
        completed_matches = completed_index.get(request.get("request_id"), [])
        # Normally one REPORT per request. Preserve all duplicate matches for audit.
        report = max(lambda_matches, key=lambda item: item.get("timestamp", 0), default={})
        access = max(api_matches, key=lambda item: item.get("timestamp", 0), default={})
        completed = max(completed_matches, key=lambda item: item.get("timestamp", 0), default={})
        model_candidates = [event for event in model_index.get(report.get("logStreamName"), [])
                            if event.get("timestamp", 0) <= report.get("timestamp", 0)]
        model = max(model_candidates, key=lambda item: item.get("timestamp", 0), default={})
        init_ms = report.get("init_duration_ms")
        row = {key: request.get(key) for key in ("run_directory", "run_id", "sequence", "phase", "iteration",
                "case_id", "started_at_utc", "status", "success", "state_chars", "question_count", "e2e_ms",
                "prediction_ms", "outside_prediction_ms", "request_id", "api_request_id",
                "first_request_in_environment", "state_tokens", "sequence_tokens")}
        row.update({"lambda_report_match_count": len(lambda_matches), "api_access_match_count": len(api_matches),
                    "lambda_duration_ms": report.get("duration_ms"),
                    "lambda_billed_duration_ms": report.get("billed_duration_ms"),
                    "lambda_memory_mb": report.get("memory_mb"),
                    "lambda_max_memory_mb": report.get("max_memory_mb"),
                    "lambda_init_duration_ms": init_ms, "lambda_report_status": report.get("status"),
                    "lambda_error_type": report.get("error_type"),
                    "lambda_log_stream": report.get("logStreamName"),
                    "initialization_evidence": "REPORT contains Init Duration" if init_ms is not None else "unknown; missing Init Duration does not rule out initialization",
                    "same_stream_preceding_model_load_ms": model.get("model_load_ms"),
                    "same_stream_preceding_model_loaded_timestamp": model.get("timestamp"),
                    "application_duration_ms": completed.get("duration_ms"),
                    "application_first_request_flag": completed.get("first_request_in_environment"),
                    "api_status": access.get("status"), "api_route": access.get("routeKey"),
                    "api_integration_latency_ms": access.get("integration_latency_ms"),
                    "api_response_length": number(access.get("responseLength")),
                    "request": request, "lambda_reports": lambda_matches,
                    "api_access_logs": api_matches, "application_completed_logs": completed_matches})
        joined.append(row)
    summary = {
        "requests": len(joined),
        "lambda_reports_matched": sum(bool(row["lambda_report_match_count"]) for row in joined),
        "lambda_reports_unmatched": sum(not row["lambda_report_match_count"] for row in joined),
        "lambda_request_id_missing": sum(not row["request_id"] for row in joined),
        "lambda_report_duplicates": sum(row["lambda_report_match_count"] > 1 for row in joined),
        "api_access_matched": sum(bool(row["api_access_match_count"]) for row in joined),
        "api_access_unmatched": sum(not row["api_access_match_count"] for row in joined),
        "api_request_id_missing": sum(not row["api_request_id"] for row in joined),
        "api_access_duplicates": sum(row["api_access_match_count"] > 1 for row in joined),
        "report_init_duration_present": sum(row["lambda_init_duration_ms"] is not None for row in joined),
        "collected_lambda_reports": len(parsed_lambda["reports"]),
        "collected_model_loaded_events": len(parsed_lambda["model_loaded"]),
        "collected_init_events": len(parsed_lambda["init_events"]),
        "collected_api_access_events": len(parsed_api),
        "unmatched_requests": [{key: row[key] for key in ("run_directory", "sequence", "case_id", "request_id",
                                "api_request_id", "lambda_report_match_count", "api_access_match_count")}
                               for row in joined if not row["lambda_report_match_count"] or not row["api_access_match_count"]],
    }
    return joined, summary


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def collect_log_events(client, log_group: str, start_ms: int, end_ms: int) -> tuple[list[dict], dict]:
    events, seen_tokens, pages = [], set(), 0
    params = {"logGroupName": log_group, "startTime": start_ms, "endTime": end_ms}
    while True:
        page = client.filter_log_events(**params)
        pages += 1
        events.extend(page.get("events", []))
        token = page.get("nextToken")
        if not token:
            break
        if token in seen_tokens:
            raise RuntimeError("CloudWatch returned a repeated pagination token; collection is incomplete")
        seen_tokens.add(token)
        params["nextToken"] = token
    # Events can be returned while logs are being ingested. Keep a stable sorted snapshot.
    events.sort(key=lambda event: (event.get("timestamp", 0), event.get("eventId", "")))
    return events, {"log_group": log_group, "pages": pages, "events": len(events),
                    "start_time_ms": start_ms, "end_time_ms": end_ms}


def safe_lambda_snapshot(client, outputs: dict) -> dict:
    function_name = outputs["FunctionName"]
    alias_name = outputs.get("FunctionAliasArn", ":live").rsplit(":", 1)[-1]
    alias = client.get_alias(FunctionName=function_name, Name=alias_name)
    configuration = client.get_function_configuration(FunctionName=function_name, Qualifier=alias["FunctionVersion"])
    fields = ("FunctionName", "FunctionArn", "Runtime", "CodeSize", "Timeout", "MemorySize", "LastModified",
              "Version", "State", "StateReasonCode", "LastUpdateStatus", "LastUpdateStatusReasonCode",
              "PackageType", "Architectures", "EphemeralStorage", "LoggingConfig", "RevisionId")
    env_fields = ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LAYA_MAX_LEN", "LAYA_HEAD_MAX_LEN",
                  "LAYA_MAX_QUESTIONS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "MODEL_DIR")
    variables = configuration.get("Environment", {}).get("Variables", {})
    concurrency = client.get_function_concurrency(FunctionName=function_name)
    try:
        provisioned = client.get_provisioned_concurrency_config(FunctionName=function_name, Qualifier=alias_name)
        provisioned_snapshot = {"configured": True, "qualifier": alias_name,
                                **{key: provisioned[key] for key in (
                                    "RequestedProvisionedConcurrentExecutions", "AvailableProvisionedConcurrentExecutions",
                                    "AllocatedProvisionedConcurrentExecutions", "Status", "StatusReason", "LastModified")
                                   if key in provisioned}}
    except Exception as exc:
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code not in ("ProvisionedConcurrencyConfigNotFoundException", "ResourceNotFoundException"):
            raise
        provisioned_snapshot = {"configured": False, "qualifier": alias_name, "not_found_code": error_code}
    return {"captured_at_utc": utc_now(), "note": "Current alias/version snapshot; not historical configuration at invocation time",
            "configuration": {key: configuration[key] for key in fields if key in configuration},
            "allowed_environment": {key: variables[key] for key in env_fields if key in variables},
            "reserved_concurrency": {"configured": "ReservedConcurrentExecutions" in concurrency,
                                     "ReservedConcurrentExecutions": concurrency.get("ReservedConcurrentExecutions")},
            "provisioned_concurrency": provisioned_snapshot,
            "alias": {key: alias[key] for key in ("AliasArn", "Name", "FunctionVersion", "RevisionId", "RoutingConfig") if key in alias}}


def run_self_test() -> int:
    class EvidenceTests(unittest.TestCase):
        def test_text_and_json_report(self):
            text = "REPORT RequestId: req-1\tDuration: 501.25 ms\tBilled Duration: 2302 ms\tMemory Size: 8192 MB\tMax Memory Used: 1900 MB\tInit Duration: 1800.5 ms\tStatus: error\tError Type: Runtime.ExitError"
            report = parse_report(text)
            self.assertEqual(report["request_id"], "req-1")
            self.assertEqual(report["duration_ms"], 501.25)
            self.assertEqual(report["init_duration_ms"], 1800.5)
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["error_type"], "Runtime.ExitError")
            platform = json.dumps({"type": "platform.report", "record": {"requestId": "req-2", "status": "success",
                "metrics": {"durationMs": 95.2, "billedDurationMs": 96, "memorySizeMB": 8192, "maxMemoryUsedMB": 2000}}})
            self.assertEqual(parse_report(platform)["duration_ms"], 95.2)
            self.assertIsNone(parse_report(platform)["init_duration_ms"])
            self.assertIsNone(parse_report("INIT_REPORT Init Duration: 100.0 ms Phase: init Status: error"))

        def test_app_api_join_and_unknown_initialization(self):
            events = [
                {"timestamp": 1000, "logStreamName": "stream", "message": '[INFO]\tdate\t\t{"event":"model_loaded","model_load_ms":750}'},
                {"timestamp": 1050, "logStreamName": "stream", "message": 'INIT_REPORT Init Duration: 800 ms Phase: init Status: success'},
                {"timestamp": 1900, "logStreamName": "stream", "message": json.dumps({"message": json.dumps({"event": "request_completed", "request_id": "req-1", "duration_ms": 102, "first_request_in_environment": True})})},
                {"timestamp": 2000, "logStreamName": "stream", "message": "REPORT RequestId: req-1\tDuration: 105 ms\tBilled Duration: 106 ms\tMemory Size: 8192 MB\tMax Memory Used: 1900 MB"},
            ]
            api = [{"timestamp": 2010, "message": '{"requestId":"api-1","status":"200","integrationLatency":"120","routeKey":"POST /predict"}'}]
            raw = [{"sequence": 1, "request_id": "req-1", "api_request_id": "api-1"}, {"sequence": 2}]
            parsed = parse_lambda_events(events)
            joined, summary = join_evidence(raw, parsed, parse_api_events(api))
            self.assertEqual(len(parsed["init_events"]), 1)
            self.assertEqual(joined[0]["same_stream_preceding_model_load_ms"], 750)
            self.assertEqual(joined[0]["application_duration_ms"], 102)
            self.assertEqual(joined[0]["api_integration_latency_ms"], 120)
            self.assertTrue(joined[0]["initialization_evidence"].startswith("unknown"))
            self.assertEqual(summary["lambda_reports_unmatched"], 1)
            self.assertEqual(summary["api_access_unmatched"], 1)
            self.assertEqual(summary["lambda_request_id_missing"], 1)

        def test_empty_page_does_not_stop_pagination(self):
            class Client:
                calls = 0
                def filter_log_events(self, **kwargs):
                    self.calls += 1
                    return {"events": [], "nextToken": "next"} if self.calls == 1 else {"events": [{"timestamp": 1, "eventId": "e"}]}
            client = Client()
            events, metadata = collect_log_events(client, "group", 0, 3000)
            self.assertEqual(client.calls, 2)
            self.assertEqual(metadata["pages"], 2)
            self.assertEqual(len(events), 1)

        def test_configuration_uses_alias_version_and_redacts(self):
            class Client:
                def get_alias(self, **kwargs):
                    return {"Name": "live", "FunctionVersion": "3", "ResponseMetadata": {"unused": True}}
                def get_function_configuration(self, **kwargs):
                    self.qualifier = kwargs["Qualifier"]
                    return {"FunctionName": "name", "MemorySize": 8192, "Environment": {"Variables": {"TORCH_NUM_THREADS": "4", "SECRET_KEY": "do-not-save"}}}
                def get_function_concurrency(self, **kwargs):
                    return {"ReservedConcurrentExecutions": 2}
                def get_provisioned_concurrency_config(self, **kwargs):
                    self.pc_qualifier = kwargs["Qualifier"]
                    return {"RequestedProvisionedConcurrentExecutions": 1, "AvailableProvisionedConcurrentExecutions": 1,
                            "AllocatedProvisionedConcurrentExecutions": 1, "Status": "READY"}
            client = Client()
            snapshot = safe_lambda_snapshot(client, {"FunctionName": "name"})
            self.assertEqual(client.qualifier, "3")
            self.assertEqual(snapshot["allowed_environment"], {"TORCH_NUM_THREADS": "4"})
            self.assertNotIn("do-not-save", json.dumps(snapshot))
            self.assertEqual(client.pc_qualifier, "live")
            self.assertEqual(snapshot["reserved_concurrency"]["ReservedConcurrentExecutions"], 2)
            self.assertEqual(snapshot["provisioned_concurrency"]["Status"], "READY")

            class NotFound(Exception):
                response = {"Error": {"Code": "ProvisionedConcurrencyConfigNotFoundException"}}
            class WithoutPc(Client):
                def get_provisioned_concurrency_config(self, **kwargs):
                    raise NotFound()
            snapshot = safe_lambda_snapshot(WithoutPc(), {"FunctionName": "name"})
            self.assertFalse(snapshot["provisioned_concurrency"]["configured"])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--stack-name", default="LayaApiStack")
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.outputs or not args.run_dir or not args.out_dir:
        parser.error("--outputs, at least one --run-dir, and --out-dir are required")
    dirs = [directory.resolve(strict=True) for directory in args.run_dir]
    if len(set(dirs)) != len(dirs):
        parser.error("Duplicate --run-dir would double count measurements")
    raw = []
    for directory in dirs:
        for line_number, line in enumerate((directory / "requests.jsonl").read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict) or not value.get("started_at_utc"):
                    raise ValueError(f"Invalid raw record: {directory}, line {line_number}")
                timestamp_ms(value["started_at_utc"])
                raw.append({**value, "run_directory": str(directory)})
    if not raw:
        raise ValueError("No request records found")
    start_ms = min(timestamp_ms(row["started_at_utc"]) for row in raw) - 60_000
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    outputs = json.loads(args.outputs.read_text(encoding="utf-8-sig"))[args.stack_name]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = ("lambda-logs.jsonl", "api-logs.jsonl", "lambda-parsed.json", "api-parsed.json", "lambda-configuration.json",
                 "joined.json", "joined.csv", "collection-summary.json")
    for filename in filenames:
        if (args.out_dir / filename).exists():
            raise ValueError(f"Refusing to overwrite previous collection: {args.out_dir / filename}")
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=args.profile, region_name=args.region or outputs["AwsRegion"])
    client_config = Config(retries={"mode": "standard", "max_attempts": 3})
    logs = session.client("logs", config=client_config)
    lambda_client = session.client("lambda", config=client_config)
    collection = {"schema_version": 1, "started_at_utc": utc_now(), "run_directories": [str(directory) for directory in dirs],
                  "region": args.region or outputs["AwsRegion"], "profile": args.profile, "start_time_ms": start_ms,
                  "end_time_ms": end_ms, "errors": [],
                  "notes": ["Read-only snapshot; logs may arrive after this collection. Recollect into a fresh directory if unmatched.",
                            "Missing REPORT Init Duration does not rule out initialization or suppressed init.",
                            "first_request_in_environment alone does not establish a cold start.",
                            "A preceding model_loaded event is associated only by log stream and time; its duration is not assigned to every request.",
                            "Request status and missing request IDs remain visible; unmatched is not proof that Lambda was never invoked."]}
    fetched = {}
    for kind, key in (("lambda", "FunctionLogGroup"), ("api", "ApiLogGroup")):
        try:
            events, metadata = collect_log_events(logs, outputs[key], start_ms, end_ms)
            fetched[kind] = events
            collection[kind] = metadata
            with (args.out_dir / f"{kind}-logs.jsonl").open("x", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as exc:
            fetched[kind] = []
            collection["errors"].append({"operation": f"collect_{kind}_logs", "type": type(exc).__name__, "message": str(exc)})
    try:
        save_json(args.out_dir / "lambda-configuration.json", safe_lambda_snapshot(lambda_client, outputs))
    except Exception as exc:
        collection["errors"].append({"operation": "read_lambda_configuration", "type": type(exc).__name__, "message": str(exc)})
    parsed_lambda = parse_lambda_events(fetched["lambda"])
    parsed_api = parse_api_events(fetched["api"])
    save_json(args.out_dir / "lambda-parsed.json", parsed_lambda)
    save_json(args.out_dir / "api-parsed.json", parsed_api)
    joined, summary = join_evidence(raw, parsed_lambda, parsed_api)
    save_json(args.out_dir / "joined.json", joined)
    nested = {"request", "lambda_reports", "api_access_logs", "application_completed_logs"}
    fields = [key for key in joined[0] if key not in nested]
    with (args.out_dir / "joined.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in joined:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key not in nested})
    collection.update({"finished_at_utc": utc_now(), "join_summary": summary})
    save_json(args.out_dir / "collection-summary.json", collection)
    print(json.dumps({"saved_to": str(args.out_dir.resolve()), **summary, "errors": collection["errors"]}, ensure_ascii=False, indent=2))
    return 1 if collection["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
