"""localhostのLambda Runtime Interface Emulatorを呼ぶ。AWS認証情報は使用しない。"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--request", type=Path, default=Path("examples/request.json"))
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1-65535")
    path, method = ("/health", "GET") if args.health else ("/predict", "POST")
    event = {"version": "2.0", "routeKey": f"{method} {path}", "rawPath": path,
             "headers": {"content-type": "application/json"},
             "requestContext": {"http": {"method": method, "path": path}},
             "isBase64Encoded": False}
    if not args.health:
        event["body"] = json.dumps(json.loads(args.request.read_text(encoding="utf-8-sig")), ensure_ascii=False)
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/2015-03-31/functions/function/invocations",
        data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if isinstance(result.get("body"), str):
        try:
            result["body"] = json.loads(result["body"])
        except json.JSONDecodeError:
            pass
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("statusCode") == 200 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
