"""CDKの設定値を検証する。文字列のcontextも扱う。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class DeploymentConfig:
    memory_mb: int = 8192
    reserved_concurrency: int = 2
    provisioned_concurrency: int = 0
    torch_threads: int = 4
    max_len: int = 1024
    head_max_len: int = 384
    max_questions: int = 8

    def __post_init__(self) -> None:
        ranges = {
            "memory_mb": (1024, 10240),
            "reserved_concurrency": (1, 50),
            "provisioned_concurrency": (0, 25),
            "torch_threads": (1, 6),
            "max_len": (256, 1024),
            "head_max_len": (64, 512),
            "max_questions": (1, 8),
        }
        for name, (lo, hi) in ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if self.head_max_len >= self.max_len - 32:
            raise ValueError("headMaxLen must leave at least 32 tokens below maxLen")
        # 新旧versionのPCが切り替わる間の枠も確保する。
        if self.provisioned_concurrency * 2 > self.reserved_concurrency:
            raise ValueError(
                "reservedConcurrency must be >= 2 * provisionedConcurrency "
                "to leave room for old/new versions during deployment"
            )

    @classmethod
    def from_context(cls, get: Callable[[str], Any]) -> "DeploymentConfig":
        fields = {
            "memory_mb": "memoryMb", "reserved_concurrency": "reservedConcurrency",
            "provisioned_concurrency": "provisionedConcurrency", "torch_threads": "torchThreads",
            "max_len": "maxLen", "head_max_len": "headMaxLen", "max_questions": "maxQuestions",
        }
        values = {}
        for field, key in fields.items():
            value = get(key)
            if value is None:
                continue
            if isinstance(value, bool) or (not isinstance(value, (str, int))):
                raise ValueError(f"{key} must be an integer")
            try:
                values[field] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc
        return cls(**values)
