from __future__ import annotations
import pytest
from errors import ApiError
from token_budget import check_token_budget
from validation import PredictionRequest


class CharTokenizer:
    mask_token = "[MASK]"
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def render(q):
    if q["t"] == "choice":
        return [f"{k}: {v}" for k,v in q["crit"].items()]
    if q["t"] == "score":
        return [f"level {i}: {v}" for i,v in enumerate(q["crit"])]
    return ["false: no", "true: yes"]


def sample(state="abc", criteria=None):
    return PredictionRequest(state, {"q": {"type":"choice", "instructions":"pick",
        "criteria": criteria or {"a":"one", "b":"two"}}})


def check(request, max_len=200, head_max_len=100):
    return check_token_budget(request, CharTokenizer(), render,
                              max_len=max_len, head_max_len=head_max_len)


def test_exact_sequence_size_and_boundary():
    stats = check(sample())
    size = 3 + len("choice question: pick") + (1+len(" a: one")) + (1+len(" b: two")) + 4
    assert stats["sequence_tokens"]["q"] == size
    check(sample(), max_len=size)
    with pytest.raises(ApiError) as error:
        check(sample(), max_len=size-1)
    assert error.value.code == "INPUT_TOO_LONG"


def test_state_too_long():
    with pytest.raises(ApiError) as error:
        check(sample(state="x"*200))
    assert error.value.status == 422


def test_option_48_token_cap():
    with pytest.raises(ApiError) as error:
        check(sample(criteria={"a":"x"*60, "b":"two"}))
    assert error.value.code == "OPTION_TOO_LONG"


def test_head_budget_overflow():
    with pytest.raises(ApiError) as error:
        check(sample(), head_max_len=20)
    assert error.value.code == "QUESTION_TOO_LONG"


def test_duplicate_tokenized_options():
    request = PredictionRequest("x", {"q":{"type":"choice", "instructions":"pick", "criteria":{"a":"x","b":"y"}}})
    with pytest.raises(ApiError) as error:
        check_token_budget(request, CharTokenizer(), lambda q:["same","same"],
                           max_len=200, head_max_len=100)
    assert error.value.code == "INDISTINGUISHABLE_OPTIONS"


def test_mask_normalization_matches_laya():
    assert check(sample("[MASK]"))["state_tokens"] == 1
