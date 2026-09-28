"""A malformed endpoint must not receive signed AWS requests."""
import importlib.util
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("api_endpoint", SCRIPTS / "api_endpoint.py")
endpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endpoint)


@pytest.mark.parametrize("region,suffix", [
    ("us-east-1", "amazonaws.com"),
    ("us-gov-west-1", "amazonaws.com"),
    ("cn-north-1", "amazonaws.com.cn"),
])
def test_regional_partitions(region, suffix):
    url = f"https://example123.execute-api.{region}.{suffix}/predict"
    assert endpoint.validate_api_endpoint(url, region) == url


@pytest.mark.parametrize("url", [
    "http://example123.execute-api.us-east-1.amazonaws.com/predict",
    "https://user@example123.execute-api.us-east-1.amazonaws.com/predict",
    "https://example123.execute-api.us-east-1.amazonaws.com/predict?query=value",
    "https://example123.execute-api.us-east-1.amazonaws.com/predict#fragment",
    "https://example123.execute-api.us-east-1.amazonaws.com/predict?",
    "https://example.invalid/predict",
    "https://example123.execute-api.us-east-1.amazonaws.com.evil.invalid/predict",
    "https://example123.execute-api.us-west-2.amazonaws.com/predict",
    "https://example123.execute-api.us-east-1.amazonaws.com:443/predict",
    "https://example123.execute-api.us-east-1.amazonaws.com/other",
    "https://example123.execute-api.us-east-1.amazonaws.com/predict\n",
    "https://example123.execute-api.us-east-1.amazonaws.com/%70redict",
    "https://example123.execute-api.cn-north-1.amazonaws.com/predict",
])
def test_unsafe_endpoint_rejected_before_credentials(monkeypatch, url):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module_spec = importlib.util.spec_from_file_location("reviews_security", SCRIPTS / "evaluate_product_reviews.py")
    reviews = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reviews)
    class NoCredentials:
        def get_credentials(self):
            pytest.fail("Credentials must not be read for an invalid endpoint")
    with pytest.raises(ValueError, match="trusted regional"):
        reviews.infer(NoCredentials(), url, "us-east-1", "test")


def test_base_url_and_route_constraints():
    base = "https://example123.execute-api.us-east-1.amazonaws.com"
    assert endpoint.validate_api_endpoint(base, "us-east-1", paths=("", "/")) == base
    with pytest.raises(ValueError):
        endpoint.validate_api_endpoint(base + "/health", "us-east-1", paths=("/predict",))
