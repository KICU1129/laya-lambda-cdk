"""HTTP入力を検証する。モデル・ネットワークに依存しない。"""
from __future__ import annotations
import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any
from errors import ApiError

MAX_BODY_BYTES = 65_536
MAX_STATE_CHARS = 12_000
MAX_OPTIONS = 20


@dataclass(frozen=True)
class PredictionRequest:
    state: str
    questions: dict[str, dict[str, Any]]


def _bad(message: str) -> None:
    raise ApiError(400, "INVALID_REQUEST", message)


def _text(value: Any, name: str, max_chars: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        _bad(f"{name} must be a string")
    if (not empty and not value.strip()) or len(value) > max_chars:
        _bad(f"{name} must be non-empty and at most {max_chars} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _bad(f"{name} must contain valid Unicode")
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _bad("Duplicate JSON keys are not allowed")
        result[key] = value
    return result


def _invalid_constant(_value):
    _bad("NaN and Infinity are not valid JSON values")


def decode_body(event: dict) -> Any:
    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Use Content-Type: application/json")
    body = event.get("body")
    if not isinstance(body, str) or not body:
        _bad("A JSON request body is required")
    # base64をデコードする前にもサイズを制限する。
    if len(body) > (MAX_BODY_BYTES * 4 // 3 + 8):
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body exceeds 64 KiB")
    encoded = event.get("isBase64Encoded", False)
    if type(encoded) is not bool:
        _bad("isBase64Encoded must be a boolean")
    try:
        raw = base64.b64decode(body, validate=True) if encoded else body.encode("utf-8")
    except (ValueError, UnicodeError, binascii.Error):
        _bad("Request body is not valid base64 or UTF-8")
    if len(raw) > MAX_BODY_BYTES:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body exceeds 64 KiB")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_pairs,
                          parse_constant=_invalid_constant)
    except (json.JSONDecodeError, UnicodeError, RecursionError, ValueError):
        _bad("Request body must be valid UTF-8 JSON")


def validate_payload(payload: Any, *, max_questions: int = 8) -> PredictionRequest:
    if not isinstance(payload, dict) or set(payload) != {"state", "questions"}:
        _bad("The body must contain exactly state and questions")
    state = _text(payload["state"], "state", MAX_STATE_CHARS)
    questions = payload["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= max_questions:
        _bad(f"questions must contain between 1 and {max_questions} entries")
    normalized = {}
    for qid, question in questions.items():
        if not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", qid):
            _bad("Question IDs must use 1-64 ASCII letters, digits, underscores, dots or hyphens")
        if not isinstance(question, dict):
            _bad("Each question must be an object")
        qtype = question.get("type")
        if qtype not in ("choice", "score", "noul"):
            _bad("Question type must be choice, score or noul")
        if not {"type", "instructions"} <= set(question):
            _bad("Each question requires type and instructions")
        allowed = {"type", "instructions", "criteria"}
        if qtype == "noul":
            allowed.add("labels")
        if set(question) - allowed:
            _bad("A question contains unsupported fields")
        instructions = _text(question["instructions"], "instructions", 1000)
        item: dict[str, Any] = {"type": qtype, "instructions": instructions}
        criteria = question.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
                _bad(f"choice criteria must contain 2-{MAX_OPTIONS} options")
            item["criteria"] = {
                _text(key, "choice label", 64): _text(value, "choice description", 512, empty=True)
                for key, value in criteria.items()
            }
        elif qtype == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                _bad("score criteria must contain 2-10 ordered descriptions")
            item["criteria"] = [_text(value, "score description", 512) for value in criteria]
        else:
            if "criteria" in question:
                if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                    _bad("noul criteria must contain exactly false and true")
                item["criteria"] = {key: _text(value, "noul description", 512)
                                    for key, value in criteria.items()}
            if "labels" in question:
                labels = question["labels"]
                if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
                    _bad("noul labels must contain exactly false and true")
                item["labels"] = {key: _text(value, "noul label", 64).strip()
                                  for key, value in labels.items()}
                if item["labels"]["false"] == item["labels"]["true"]:
                    _bad("noul labels must be distinct")
        normalized[qid] = item
    return PredictionRequest(state=state, questions=normalized)
