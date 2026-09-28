"""IAM SigV4署名を付けてLaya APIを呼び出す。AWS SSO profileも利用可能。"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_outputs(path: Path, stack_name: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if stack_name not in data:
        raise ValueError(f"Stack {stack_name!r} not found in {path}")
    return data[stack_name]


def main() -> int:
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("cdk-outputs.json"))
    parser.add_argument("--stack-name", default="LayaApiStack")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--path", choices=["/predict", "/health"], default="/predict")
    parser.add_argument("--request", type=Path, default=Path("examples/request.json"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.6)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or args.interval < 0:
        parser.error("repeat must be 1-100 and interval must be >= 0")
    outputs = load_outputs(args.outputs, args.stack_name)
    region = args.region or outputs["AwsRegion"]
    url = outputs["ApiUrl"].rstrip("/") + args.path
    from api_endpoint import validate_api_endpoint
    validate_api_endpoint(outputs["ApiUrl"], region, paths=("", "/"))
    validate_api_endpoint(url, region, paths=(args.path,))
    session = boto3.Session(profile_name=args.profile, region_name=region)
    method = "GET" if args.path == "/health" else "POST"
    payload = None
    if method == "POST":
        # UTF-8 BOMの有無やOSの改行に依存しないバイト列を署名・送信する。
        data = json.loads(args.request.read_text(encoding="utf-8-sig"))
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    opener = urllib.request.build_opener(NoRedirect())
    times = []
    failures = 0
    for i in range(args.repeat):
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials were not found; configure a profile or run aws sso login")
        request = AWSRequest(method=method, url=url, data=payload,
                             headers={"Content-Type": "application/json", "Accept": "application/json"})
        SigV4Auth(credentials.get_frozen_credentials(), "execute-api", region).add_auth(request)
        signed = urllib.request.Request(url, data=payload, method=method, headers=dict(request.headers))
        started = time.perf_counter()
        try:
            with opener.open(signed, timeout=35) as response:
                status, body = response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            print(f"Network error: {exc.reason}", file=sys.stderr)
            return 1
        elapsed = (time.perf_counter() - started) * 1000
        times.append(elapsed)
        failures += int(status >= 400)
        print(f"Request {i+1}: HTTP {status}, client elapsed {elapsed:.1f} ms")
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(body)
        if i + 1 < args.repeat:
            time.sleep(args.interval)
    if len(times) > 1:
        print(json.dumps({"requests": len(times), "failures": failures,
                          "client_latency_ms": {"min": round(min(times), 1),
                                                "median": round(statistics.median(times), 1),
                                                "max": round(max(times), 1)}}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
