"""AWS Lambdaハンドラー。モデルロードはInitフェーズで一度だけ実施する。"""
from __future__ import annotations
import logging
import os
from inference import LayaEngine
from service import ApiService

logging.getLogger().setLevel(os.environ.get("LOG_LEVEL", "INFO"))
_ENGINE = LayaEngine()
_SERVICE = ApiService(_ENGINE)


def lambda_handler(event, context):
    """IAM認証後のAPI Gateway HTTP API v2イベントを処理する。"""
    return _SERVICE.handle(event, context)
