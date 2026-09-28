"""Layaモデルを一度ロードし、CPUで推論する。"""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from pathlib import Path
from validation import PredictionRequest
from token_budget import check_token_budget

LOG = logging.getLogger(__name__)


class LayaEngine:
    def __init__(self) -> None:
        started = time.perf_counter()
        # 未設定時もオフラインを既定とする。MODEL_DIRからのみロードする。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HOME", "/tmp/huggingface")
        import torch
        import laya
        from laya.common import render_options
        if laya.__version__ != "0.3.21":
            raise RuntimeError("Revalidate token-budget checks before changing Laya version")
        threads = int(os.environ.get("TORCH_NUM_THREADS", "4"))
        if not 1 <= threads <= 6:
            raise ValueError("TORCH_NUM_THREADS must be between 1 and 6")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        self.max_len = int(os.environ.get("LAYA_MAX_LEN", "1024"))
        self.head_max_len = int(os.environ.get("LAYA_HEAD_MAX_LEN", "384"))
        self.max_questions = int(os.environ.get("LAYA_MAX_QUESTIONS", "8"))
        if not 256 <= self.max_len <= 1024 or not 64 <= self.head_max_len < self.max_len - 32:
            raise ValueError("Invalid Laya token-budget configuration")
        if not 1 <= self.max_questions <= 8:
            raise ValueError("LAYA_MAX_QUESTIONS must be between 1 and 8")
        root = Path(os.environ.get("MODEL_DIR", "/opt/laya-model")).resolve(strict=True)
        manifest = json.loads((root / "build-manifest.json").read_text(encoding="utf-8"))
        # 毎回の重み全量ハッシュはcold startを増やすため、build smoke testで検証。
        self.agent = laya.Agent(str(root), device="cpu", fast=False, compile=False)
        self._render_options = render_options
        self._lock = threading.Lock()
        self.info = {
            "model": manifest["repo_id"], "model_revision": manifest["revision"],
            "laya_version": laya.__version__, "device": "cpu", "torch_threads": threads,
            "max_len": self.max_len, "head_max_len": self.head_max_len,
            "max_questions": self.max_questions,
        }
        LOG.info(json.dumps({"event": "model_loaded", **self.info,
                             "model_load_ms": round((time.perf_counter()-started)*1000, 2)}))

    def predict(self, request: PredictionRequest) -> tuple[dict, dict]:
        # Lambdaは1実行環境で通常1リクエストだが、tokenizerの共有を明示的に直列化。
        with self._lock:
            tokens = check_token_budget(request, self.agent.tok, self._render_options,
                                        max_len=self.max_len, head_max_len=self.head_max_len)
            result = self.agent.predict(request.state, request.questions,
                                        max_len=self.max_len, head_max_len=self.head_max_len)
        return result, tokens
