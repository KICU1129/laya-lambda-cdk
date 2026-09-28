"""Accuracy reporting: an incorrect positive must remain visible in every aggregate."""
import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "scripts/evaluate_accuracy.py"
spec = importlib.util.spec_from_file_location("evaluate_accuracy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def answers(refund_probability):
    return {
        "department": {"choice": "billing", "probabilities": {"billing": 1.0}},
        "urgency": {"score": 1.0, "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}},
        "refund_requested": {"noul": refund_probability},
        "duplicate_charge": {"noul": 0.01},
        "cancellation_requested": {"noul": 0.01},
        "service_unavailable": {"noul": 0.01},
        "contact_channel": {"choice": "unspecified", "probabilities": {"unspecified": 0.9}},
        "sentiment": {"score": 0.0, "probabilities": {"0": 0.9, "1": 0.08, "2": 0.02}},
    }


def test_false_positive_reduces_precision_and_exact_match():
    fixture = {"cases": []}
    records = []
    for case_id, gold_refund in (("positive", True), ("negative", False)):
        gold = {"department": "billing", "urgency": 1,
                "refund_requested": gold_refund, "duplicate_charge": False,
                "cancellation_requested": False, "service_unavailable": False,
                "contact_channel": "unspecified", "sentiment": 0}
        fixture["cases"].append({"id": case_id, "state": "評価用の問い合わせ", "gold": gold})
        records.append({"id": case_id, "success": True,
                        "response": {"result": {"answers": answers(0.9)}}})
    rows, result = module.score(records, fixture)
    refund = result["by_question"]["refund_requested"]
    assert (refund["tp"], refund["fp"], refund["tn"], refund["fn"]) == (1, 1, 0, 0)
    assert refund["precision"] == 0.5
    assert refund["recall"] == 1.0
    assert refund["f1"] == 0.6667
    assert result["micro_accuracy"] == 0.9375
    assert result["exact_match_all_8"]["correct"] == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == "negative"
    assert len(rows) == 16


def test_score_category_comes_from_distribution_not_rounded_mean():
    answer = {"score": 1.49, "probabilities": {"0": 0.05, "1": 0.46, "2": 0.49}}
    assert module.predicted(answer, "urgency") == (2, 0.49)


def test_existing_ja_fixture_is_valid():
    fixture = module.load_fixture(Path(__file__).resolve().parents[1] / "examples/accuracy-ja.json")
    assert len(fixture["cases"]) >= 36
