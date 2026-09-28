import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sanitize_review_results import project_record


def test_unknown_environment_and_error_details_never_pass_projection():
    record = {"id": "R001", "success": False, "predicted": None, "correct": False,
              "client_ms": 10, "prediction_ms": None, "region": "private-region",
              "started_at_utc": "private-time", "client_environment": {"machine": "private"},
              "error": "private diagnostic", "unexpected_future_secret": "private-value"}
    public = project_record(record)
    assert public == {"id": "R001", "success": False, "predicted": None, "correct": False,
                      "client_ms": 10, "prediction_ms": None, "error": "request_failed"}


def test_unrecognized_nested_metadata_does_not_escape():
    record = {"success": True, "usage": {"input_tokens": 1, "machine": "private"},
              "sequence_tokens": {"sentiment": 2, "endpoint": "private"},
              "probabilities": {"positive": 1.0, "private_field": "private"}}
    assert project_record(record) == {
        "success": True, "error": None, "usage": {"input_tokens": 1},
        "sequence_tokens": {"sentiment": 2}, "probabilities": {"positive": 1.0}}
