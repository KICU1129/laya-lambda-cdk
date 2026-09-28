"""Laya 0.3.21の入力形式に合わせ、黙ったトークン切り捨てを防ぐ。"""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
from errors import ApiError
from validation import PredictionRequest


def check_token_budget(request: PredictionRequest, tokenizer: Any,
                       render_options: Callable, *, max_len: int,
                       head_max_len: int) -> dict:
    """上限超過は422。Laya側の48-token option capより前に検査する。"""
    mask_token = tokenizer.mask_token
    if not isinstance(mask_token, str) or not mask_token:
        raise RuntimeError("The checkpoint tokenizer has no mask token")

    def encode(text: str) -> list[int]:
        return tokenizer(text.replace(mask_token, " "), add_special_tokens=False,
                         truncation=False)["input_ids"]

    state_ids = encode(request.state)
    lengths = {}
    for qid, q in request.questions.items():
        internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
        if "labels" in q:
            internal["labels"] = q["labels"]
        option_tokens = [encode(" " + text) for text in render_options(internal)]
        if any(len(ids) > 48 for ids in option_tokens):
            raise ApiError(422, "OPTION_TOO_LONG",
                           f"Question {qid}: each rendered option must fit in 48 tokens")
        # build_sequenceと同じ: 各optionに[MASK]、全体にCLS/SEP/SEP/SEP。
        option_size = sum(1 + len(ids) for ids in option_tokens)
        instruction_ids = encode(f"{q['type']} question: {q['instructions']}")
        instruction_budget = head_max_len - option_size
        if instruction_budget < 16 or len(instruction_ids) > max(8, instruction_budget):
            raise ApiError(422, "QUESTION_TOO_LONG",
                           f"Question {qid}: instructions/options exceed the head token budget")
        if len({tuple(ids) for ids in option_tokens}) != len(option_tokens):
            raise ApiError(422, "INDISTINGUISHABLE_OPTIONS",
                           f"Question {qid}: options become identical after tokenization")
        length = len(state_ids) + len(instruction_ids) + option_size + 4
        if length > max_len:
            raise ApiError(422, "INPUT_TOO_LONG",
                           f"Question {qid}: state + question need {length} tokens; limit is {max_len}")
        lengths[qid] = length
    return {"state_tokens": len(state_ids), "sequence_tokens": lengths}
