"""Validate the default API Gateway endpoint before accessing signing credentials."""
from __future__ import annotations
import re
from urllib.parse import urlsplit


def validate_api_endpoint(url: str, region: str, *, paths=("/predict", "/health")) -> str:
    """Allow only this project's regional execute-api HTTPS endpoint format.

    Custom domains require an explicit separately reviewed allowlist. An AWS
    hostname alone is not proof of ownership; use only trusted local CDK outputs.
    """
    message = "Expected a trusted regional API Gateway HTTPS endpoint and allowed route"
    if not isinstance(url, str) or not isinstance(region, str):
        raise ValueError(message)
    if any(ord(char) <= 32 or ord(char) == 127 for char in url) or "?" in url or "#" in url:
        raise ValueError(message)
    suffix = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    if not re.fullmatch(r"(?:[a-z]{2}|us-gov)-[a-z]+-\d+", region):
        raise ValueError(message)
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        valid = (parsed.scheme == "https" and parsed.username is None and parsed.password is None
                 and parsed.port is None and parsed.path in paths
                 and re.fullmatch(r"[a-z0-9]{10}\.execute-api\." + re.escape(region)
                                  + r"\." + re.escape(suffix), host)
                 and parsed.netloc == host)
    except ValueError:
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)
    return url
