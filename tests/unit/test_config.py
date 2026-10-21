import pytest

from hotel_rag.config import Settings, load_yaml_defaults


def _write_yaml(tmp_path, text: str):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "llm.yaml").write_text(text, encoding="utf-8")


def test_yaml_file_provides_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("LLM_MODEL", "LLM_BASE_URL", "LLM_TEMPERATURE", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    _write_yaml(
        tmp_path,
        "base_url: http://yaml:1234/v1\nmodel: yaml-model\ntemperature: 0.4\ntimeout_seconds: 12\n",
    )

    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "http://yaml:1234/v1"
    assert settings.llm_model == "yaml-model"
    assert settings.llm_temperature == 0.4
    assert settings.llm_timeout_seconds == 12


def test_environment_variables_override_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path, "model: yaml-model\n")
    monkeypatch.setenv("LLM_MODEL", "env-model")

    assert Settings(_env_file=None).llm_model == "env-model"


def test_missing_yaml_file_keeps_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert Settings(_env_file=None).llm_model == ""


def test_api_key_cannot_live_in_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_yaml(tmp_path, "api_key: leaked-secret\n")

    with pytest.raises(ValueError, match="api_key"):
        load_yaml_defaults(tmp_path / "config" / "llm.yaml")


def test_unknown_yaml_key_fails_fast(tmp_path):
    _write_yaml(tmp_path, "modle: typo\n")

    with pytest.raises(ValueError, match="modle"):
        load_yaml_defaults(tmp_path / "config" / "llm.yaml")
