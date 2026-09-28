"""ビルド中に実モデルをロードし、非root・オフラインで推論を確認する。"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    root = Path(os.environ["MODEL_DIR"])
    # Lambdaの最小権限ユーザーに近づける。rootの権限で成功させない。
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    manifest = json.loads((root / "build-manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, f"Model artifact changed: {name}"
    if os.access(root / "tokenizer/tokenizer_config.json", os.W_OK):
        raise RuntimeError("Smoke test must use a read-only model directory")

    # offline envだけでなくPython socket接続自体を拒否し、隠れた取得を検知する。
    def reject_network(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Network access is forbidden during the offline smoke test")
    sys.addaudithook(reject_network)
    sys.path.insert(0, os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
    import handler
    from token_budget import check_token_budget
    from validation import validate_payload
    from laya.common import build_sequence
    body = {
        "state": "同じ請求が二重に引かれています。返金をお願いします。",
        "questions": {
            "department": {"type": "choice", "instructions": "担当部署を選んでください。",
                           "criteria": {"billing": "請求や返金", "technical": "不具合や操作", "other": "それ以外"}},
            "urgency": {"type": "score", "instructions": "緊急度を評価してください。", "criteria": ["低い", "通常", "高い"]},
            "refund": {"type": "noul", "instructions": "返金を求めていますか？"},
        },
    }
    context = SimpleNamespace(aws_request_id="build-smoke",
                              get_remaining_time_in_millis=lambda: 120_000)
    event = {"version": "2.0", "rawPath": "/predict", "requestContext": {"http": {"method": "POST"}},
             "headers": {"content-type": "application/json"}, "body": json.dumps(body, ensure_ascii=False)}
    response = handler.lambda_handler(event, context)
    assert response["statusCode"] == 200, response
    parsed = json.loads(response["body"])
    assert set(parsed["result"]["answers"]) == set(body["questions"]), parsed
    # ガードの計算と、固定したLayaバージョンの実際のsequence長を突き合わせる。
    engine = handler._ENGINE
    request = validate_payload(body)
    tokens = check_token_budget(request, engine.agent.tok, engine._render_options,
                                max_len=engine.max_len, head_max_len=engine.head_max_len)
    for qid, q in request.questions.items():
        internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
        ids, _ = build_sequence(engine.agent.tok, request.state, internal,
                                max_len=engine.max_len, head_max_len=engine.head_max_len)
        assert len(ids) == tokens["sequence_tokens"][qid], "Token-budget implementation drift"
    # 大きすぎる入力を推論前に拒否できることも実tokenizerで確認する。
    body["state"] = "これはテスト用の長い文章です。" * 300
    event["body"] = json.dumps(body, ensure_ascii=False)
    rejected = handler.lambda_handler(event, context)
    assert rejected["statusCode"] == 422, rejected
    print(json.dumps({"build_smoke_test": "passed", "model": engine.info["model"],
                      "revision": engine.info["model_revision"]}))


if __name__ == "__main__":
    main()
