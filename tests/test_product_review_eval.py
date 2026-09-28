import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("reviews", Path(__file__).resolve().parents[1] / "scripts/evaluate_product_reviews.py")
reviews = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviews)


def test_failures_and_missing_results_remain_in_accuracy_denominator():
    cases = [{"id": "a", "expected": "positive"}, {"id": "b", "expected": "neutral"},
             {"id": "c", "expected": "negative"}, {"id": "d", "expected": "negative"}]
    rows = [{"id": "a", "success": True, "predicted": "positive", "client_ms": 100, "prediction_ms": 80},
            {"id": "b", "success": True, "predicted": "positive", "client_ms": 200, "prediction_ms": 180},
            {"id": "c", "success": False, "predicted": None, "client_ms": 30000, "prediction_ms": None}]
    result = reviews.summarize(cases, rows)
    assert result["accuracy"] == 0.25
    assert result["successful_only_accuracy"] == 0.5
    assert result["failed_or_missing"] == 2
    assert result["confusion"]["negative"]["error"] == 2
    assert result["per_class"]["positive"]["precision"] == 0.5
    assert result["per_class"]["positive"]["recall"] == 1
    assert abs(result["macro_f1"] - 2/9) < 1e-12
    assert result["client_latency_ms"]["p50"] == 150
    assert result["client_latency_ms"]["p95"] == 195
    assert result["all_attempt_latency_ms"]["max"] == 30000


def test_duplicate_results_cannot_inflate_score():
    import pytest
    with pytest.raises(ValueError, match="repeated"):
        reviews.summarize([{"id": "a", "expected": "positive"}], [{"id": "a"}, {"id": "a"}])
