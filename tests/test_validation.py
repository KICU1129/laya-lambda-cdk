from __future__ import annotations
import base64
import json
from pathlib import Path
import pytest
from errors import ApiError
from validation import decode_body, validate_payload


@pytest.fixture
def sample():
    return json.loads((Path(__file__).resolve().parents[1]/"examples/request.json").read_text())


def event(body, **kwargs):
    return {"headers": {"content-type": "application/json"}, "body": body, **kwargs}


def test_all_types(sample):
    parsed = validate_payload(sample)
    assert parsed.state == sample["state"]
    assert {q["type"] for q in parsed.questions.values()} == {"choice", "score", "noul"}


@pytest.mark.parametrize("body", ["{", "null", "[]", '"text"', '{"state":"x"}'])
def test_bad_payload(body):
    with pytest.raises(ApiError):
        validate_payload(decode_body(event(body)))


@pytest.mark.parametrize("field,value", [("state", ""), ("state", 7), ("state", "a"*12001),
                                        ("state", "\ud800"), ("questions", {}), ("questions", [])])
def test_bad_fields(sample, field, value):
    sample[field] = value
    with pytest.raises(ApiError):
        validate_payload(sample)


def test_rejects_arbitrary_model_or_remote_url(sample):
    sample["model"] = "https://example.invalid/model"
    with pytest.raises(ApiError):
        validate_payload(sample)


def test_question_limit(sample):
    q = sample["questions"]["refund_requested"]
    sample["questions"] = {f"q{i}": q for i in range(9)}
    with pytest.raises(ApiError):
        validate_payload(sample)


@pytest.mark.parametrize("kind,criteria", [("choice", {"a":"one"}), ("choice", ["a","b"]),
    ("choice", {str(i):"desc" for i in range(21)}), ("score", ["a"]),
    ("score", {"a":"x","b":"y"}), ("noul", {"true":"yes"}), ("noul", None)])
def test_invalid_criteria(sample, kind, criteria):
    sample["questions"] = {"q": {"type": kind, "instructions": "test", "criteria": criteria}}
    with pytest.raises(ApiError):
        validate_payload(sample)


def test_noul_labels(sample):
    sample["questions"] = {"q": {"type":"noul", "instructions":"test",
        "criteria":{"false":"no", "true":"yes"}, "labels":{"false":"B", "true":"A"}}}
    assert validate_payload(sample).questions["q"]["labels"]["true"] == "A"
    sample["questions"]["q"]["labels"]["false"] = "A"
    with pytest.raises(ApiError):
        validate_payload(sample)


def test_duplicate_keys():
    with pytest.raises(ApiError):
        decode_body(event('{"state":"one","state":"two","questions":{}}'))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numbers(constant):
    with pytest.raises(ApiError):
        decode_body(event('{"state":'+constant+'}'))


def test_base64_and_uppercase_content_type(sample):
    encoded = base64.b64encode(json.dumps(sample,ensure_ascii=False).encode()).decode()
    assert decode_body({"headers":{"Content-Type":"application/json; charset=utf-8"},
                        "body":encoded,"isBase64Encoded":True}) == sample


def test_bad_base64():
    with pytest.raises(ApiError):
        decode_body(event("@@@", isBase64Encoded=True))


def test_utf8_bytes_limit():
    with pytest.raises(ApiError) as error:
        decode_body(event("あ" * 30000))
    assert error.value.status == 413


def test_content_type_required():
    with pytest.raises(ApiError) as error:
        decode_body({"body":"{}"})
    assert error.value.status == 415


def test_question_ids_are_bounded(sample):
    sample["questions"] = {"q"*65: {"type":"noul", "instructions":"test"}}
    with pytest.raises(ApiError):
        validate_payload(sample)


def test_unknown_question_fields(sample):
    sample["questions"]["refund_requested"]["temperature"] = 0
    with pytest.raises(ApiError):
        validate_payload(sample)
