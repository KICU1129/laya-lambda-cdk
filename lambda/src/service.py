"""HTTP API v2イベントを処理する。テスト時は推論エンジンを注入できる。"""
from __future__ import annotations
import json
import logging
import time
import uuid
from typing import Any, Protocol
from errors import ApiError
from validation import PredictionRequest, decode_body, validate_payload

LOG = logging.getLogger(__name__)


class Engine(Protocol):
    info: dict
    max_questions: int
    def predict(self, request: PredictionRequest) -> tuple[dict, dict]: ...


class ApiService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.first_request = True

    def handle(self, event: dict, context: Any) -> dict:
        request_id = getattr(context, "aws_request_id", None) or str(uuid.uuid4())
        first_request, self.first_request = self.first_request, False
        started = time.perf_counter()
        status = 500
        try:
            if not isinstance(event, dict) or event.get("version") != "2.0":
                raise ApiError(400, "INVALID_EVENT", "An API Gateway HTTP API v2 event is required")
            http = (event.get("requestContext") or {}).get("http") or {}
            method = http.get("method", "")
            path = event.get("rawPath", "")
            if (method, path) == ("GET", "/health"):
                response = self._response(200, request_id, {"status": "ready", **self.engine.info})
            elif (method, path) == ("POST", "/predict"):
                remaining = getattr(context, "get_remaining_time_in_millis", lambda: 60_000)()
                if remaining < 2000:
                    raise ApiError(503, "INSUFFICIENT_TIME", "Retry this request later")
                request = validate_payload(decode_body(event), max_questions=self.engine.max_questions)
                infer_started = time.perf_counter()
                result, token_info = self.engine.predict(request)
                response = self._response(200, request_id, {
                    "result": result,
                    "meta": {
                        "model": self.engine.info["model"],
                        "model_revision": self.engine.info["model_revision"],
                        "laya_version": self.engine.info["laya_version"],
                        "prediction_ms": round((time.perf_counter()-infer_started)*1000, 2),
                        "first_request_in_environment": first_request,
                        **token_info,
                    },
                })
            elif path in ("/health", "/predict"):
                raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed")
            else:
                raise ApiError(404, "NOT_FOUND", "Route not found")
            status = response["statusCode"]
            return response
        except ApiError as exc:
            status = exc.status
            return self._response(status, request_id, {"error": {"code": exc.code, "message": exc.message}})
        except Exception as exc:
            # 入力本文や例外メッセージに個人情報が混じる可能性があるので記録しない。
            LOG.error(json.dumps({"event": "inference_failed", "request_id": request_id,
                                  "error_type": type(exc).__name__}))
            status = 500
            return self._response(500, request_id, {"error": {
                "code": "INTERNAL_ERROR", "message": "Inference failed; check logs using request_id"}})
        finally:
            LOG.info(json.dumps({"event": "request_completed", "request_id": request_id,
                                 "status": status, "duration_ms": round((time.perf_counter()-started)*1000, 2),
                                 "first_request_in_environment": first_request}))

    @staticmethod
    def _response(status: int, request_id: str, payload: dict) -> dict:
        return {
            "statusCode": status,
            "headers": {"Content-Type": "application/json; charset=utf-8",
                        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            "isBase64Encoded": False,
            "body": json.dumps({"request_id": request_id, **payload}, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":")),
        }
