"""AWSを呼ばず、計測の失敗記録・標本分離・ケース設計を検証する。"""
from __future__ import annotations

from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Response(io.BytesIO):
    status = 200
    headers = {"Apigw-Requestid": "api-id"}


class Opener:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class BenchmarkTests(unittest.TestCase):
    def test_fixture_and_schedule(self):
        cases = benchmark.load_cases(ROOT / "examples" / "benchmark-ja.json")
        self.assertEqual(len(cases), 9)
        self.assertEqual({case["question_count"] for case in cases}, {1, 3, 8})
        self.assertLess(cases[0]["state_chars"], cases[3]["state_chars"])
        self.assertLess(cases[3]["state_chars"], cases[6]["state_chars"])
        scheduled = benchmark.make_schedule(cases, 1, 3, 42)
        self.assertEqual(len(scheduled), 36)
        self.assertEqual(scheduled, benchmark.make_schedule(cases, 1, 3, 42))
        for iteration in range(1, 4):
            measured = [case["id"] for phase, step, case in scheduled
                        if phase == "measurement" and step == iteration]
            self.assertEqual(set(measured), {case["id"] for case in cases})
        with self.assertRaisesRegex(ValueError, "Unknown case"):
            benchmark.load_cases(ROOT / "examples" / "benchmark-ja.json", ["typo"])

    def test_success_timing_and_request_ids(self):
        response = {"request_id": "lambda-id", "result": {"answers": {}},
                    "meta": {"prediction_ms": 125.5, "first_request_in_environment": True,
                             "state_tokens": 20, "sequence_tokens": {"a": 40}}}
        opener = Opener(Response(json.dumps(response).encode()))
        with patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.2]):
            row = benchmark.call_api(opener, object(), 35)
        self.assertEqual(opener.calls, 1)
        self.assertTrue(row["success"])
        self.assertAlmostEqual(row["e2e_ms"], 200)
        self.assertAlmostEqual(row["outside_prediction_ms"], 74.5)
        self.assertEqual(row["request_id"], "lambda-id")
        self.assertEqual(row["api_request_id"], "api-id")
        self.assertEqual(row["sequence_tokens"], {"a": 40})

    def test_http_error_saved_without_retry(self):
        headers = Message()
        headers["Apigw-Requestid"] = "error-id"
        opener = Opener(urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests",
                                               headers, io.BytesIO(b'{"message":"throttled"}')))
        row = benchmark.call_api(opener, object(), 35)
        self.assertEqual(opener.calls, 1)
        self.assertFalse(row["success"])
        self.assertEqual(row["status"], 429)
        self.assertEqual(row["api_request_id"], "error-id")
        self.assertEqual(row["response_json"], {"message": "throttled"})

    def test_network_error_saved_without_retry(self):
        opener = Opener(urllib.error.URLError(TimeoutError("socket timed out")))
        row = benchmark.call_api(opener, object(), 35)
        self.assertEqual(opener.calls, 1)
        self.assertFalse(row["success"])
        self.assertIsNone(row["status"])
        self.assertEqual(row["transport_error"]["type"], "URLError")
        self.assertGreaterEqual(row["e2e_ms"], 0)

    def test_bad_http_200_is_not_counted_as_success(self):
        row = benchmark.call_api(Opener(Response(b'{"message":"not prediction output"}')), object(), 35)
        self.assertFalse(row["success"])
        self.assertIsNotNone(row["response_contract_error"])

    def test_warmup_failure_and_first_flag_are_separated(self):
        cases = benchmark.load_cases(ROOT / "examples" / "benchmark-ja.json", ["short-q1"])
        common = {"case_id": "short-q1", "phase": "measurement", "success": True,
                  "prediction_ms": 10, "outside_prediction_ms": 20,
                  "first_request_in_environment": False}
        rows = [{**common, "sequence": 1, "phase": "warmup", "e2e_ms": 10000,
                 "first_request_in_environment": True},
                {**common, "sequence": 2, "e2e_ms": 100},
                {**common, "sequence": 3, "e2e_ms": 200},
                {**common, "sequence": 4, "success": False, "e2e_ms": 35000,
                 "prediction_ms": None, "outside_prediction_ms": None},
                {**common, "sequence": 5, "e2e_ms": 2000, "first_request_in_environment": True}]
        summary = benchmark.summarize(rows, cases, {})
        group = summary["by_case"]["short-q1"]
        self.assertEqual(group["all_measurements_including_failures"]["requests"], 4)
        self.assertEqual(group["successful_measurements"]["requests"], 3)
        steady = group["successful_measurements_first_flag_false"]
        self.assertEqual(steady["requests"], 2)
        self.assertEqual(steady["e2e_ms"]["p50"], 150)
        self.assertEqual(steady["e2e_ms"]["p95"], 195)
        self.assertEqual(summary["first_flag_true_requests"], [1, 5])
        self.assertEqual(summary["total_failures"], 1)


if __name__ == "__main__":
    unittest.main()
