"""API呼び出し元に付ける最小IAMポリシーを表示する。IAM自体は変更しない。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from invoke import load_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("cdk-outputs.json"))
    parser.add_argument("--stack-name", default="LayaApiStack")
    args = parser.parse_args()
    outputs = load_outputs(args.outputs, args.stack_name)
    print(json.dumps({"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Action": "execute-api:Invoke",
        "Resource": [outputs["PredictInvokeArn"], outputs["HealthInvokeArn"]],
    }]}, indent=2))


if __name__ == "__main__":
    main()
