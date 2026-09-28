from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from errors import ApiError
from service import ApiService


class FakeEngine:
    info = {"model":"test-model", "model_revision":"test-revision", "laya_version":"test"}
    max_questions = 8
    calls = 0
    def predict(self, request):
        self.calls += 1
        return {"answers": {key: {"type": value["type"]} for key, value in request.questions.items()}}, {"state_tokens":10}


def make_event(path="/predict", method="POST"):
    sample = json.loads((Path(__file__).resolve().parents[1]/"examples/request.json").read_text())
    return {"version":"2.0", "rawPath":path, "requestContext":{"http":{"method":method}},
            "headers":{"content-type":"application/json"}, "body":json.dumps(sample)}


CTX = SimpleNamespace(aws_request_id="test-request", get_remaining_time_in_millis=lambda:20000)


def test_predict_and_reuse():
    engine = FakeEngine()
    service = ApiService(engine)
    first = service.handle(make_event(), CTX)
    second = service.handle(make_event(), CTX)
    assert first["statusCode"] == 200
    assert json.loads(first["body"])["meta"]["first_request_in_environment"] is True
    assert json.loads(second["body"])["meta"]["first_request_in_environment"] is False
    assert engine.calls == 2
    assert first["headers"]["Cache-Control"] == "no-store"


def test_health_does_not_infer():
    engine = FakeEngine()
    result = ApiService(engine).handle(make_event("/health", "GET"), CTX)
    assert result["statusCode"] == 200
    assert json.loads(result["body"])["status"] == "ready"
    assert engine.calls == 0


def test_invalid_input_does_not_infer():
    engine = FakeEngine()
    event = make_event(); event["body"] = "{}"
    result = ApiService(engine).handle(event, CTX)
    assert result["statusCode"] == 400
    assert engine.calls == 0


def test_internal_errors_do_not_leak_input(caplog):
    class Broken(FakeEngine):
        def predict(self, request):
            raise RuntimeError("PRIVATE-CONTENT-DO-NOT-LOG")
    result = ApiService(Broken()).handle(make_event(), CTX)
    assert result["statusCode"] == 500
    assert "PRIVATE-CONTENT" not in result["body"]
    assert "PRIVATE-CONTENT" not in caplog.text


def test_budget_error_is_422():
    class Oversize(FakeEngine):
        def predict(self, request):
            raise ApiError(422, "INPUT_TOO_LONG", "too long")
    result = ApiService(Oversize()).handle(make_event(), CTX)
    assert result["statusCode"] == 422


def test_invalid_event():
    assert ApiService(FakeEngine()).handle({}, CTX)["statusCode"] == 400


def test_missing_route():
    assert ApiService(FakeEngine()).handle(make_event("/unknown"), CTX)["statusCode"] == 404


def test_wrong_method():
    assert ApiService(FakeEngine()).handle(make_event(method="GET"), CTX)["statusCode"] == 405


def test_not_enough_execution_time():
    context = SimpleNamespace(aws_request_id="test", get_remaining_time_in_millis=lambda:100)
    assert ApiService(FakeEngine()).handle(make_event(), context)["statusCode"] == 503


def test_non_finite_model_output_does_not_escape():
    class Invalid(FakeEngine):
        def predict(self, request):
            return {"bad":float("nan")}, {}
    assert ApiService(Invalid()).handle(make_event(), CTX)["statusCode"] == 500
