import pytest
from infra.config import DeploymentConfig


def test_defaults():
    assert DeploymentConfig().memory_mb == 8192
    assert DeploymentConfig().provisioned_concurrency == 0


def test_string_context():
    values = {"memoryMb":"10240", "provisionedConcurrency":"1", "reservedConcurrency":"2"}
    cfg = DeploymentConfig.from_context(values.get)
    assert cfg.memory_mb == 10240 and cfg.provisioned_concurrency == 1


@pytest.mark.parametrize("kwargs", [
    {"memory_mb":100}, {"memory_mb":10241}, {"memory_mb":True},
    {"reserved_concurrency":0}, {"reserved_concurrency":1,"provisioned_concurrency":1},
    {"torch_threads":0}, {"max_len":8192}, {"max_len":256,"head_max_len":256},
    {"max_questions":9},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DeploymentConfig(**kwargs)


@pytest.mark.parametrize("value", [True, 1.5, "1.5", "abc"])
def test_invalid_context(value):
    with pytest.raises(ValueError):
        DeploymentConfig.from_context({"memoryMb":value}.get)
