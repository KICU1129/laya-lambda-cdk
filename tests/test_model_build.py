import importlib.util
import json
from pathlib import Path


def loader():
    path = Path(__file__).resolve().parents[1]/"lambda/build/download_model.py"
    spec = importlib.util.spec_from_file_location("download_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenizer_patch_is_idempotent(tmp_path):
    path = tmp_path/"tokenizer_config.json"
    path.write_text(json.dumps({"tokenizer_class":"TokenizersBackend", "backend":"tokenizers",
                               "is_local":True, "extra_special_tokens":["<x>","<y>"]}))
    patch = loader().normalize_tokenizer_config
    assert patch(path) is True
    assert patch(path) is False
    result = json.loads(path.read_text())
    assert result["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert result["extra_special_tokens"] == {"extra_0":"<x>","extra_1":"<y>"}
    assert "backend" not in result


def test_model_revision_is_not_mutable():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/"lambda/model.lock.json").read_text())
    assert len(lock["revision"]) == 40
    assert set(lock["revision"]) <= set("0123456789abcdef")
