from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_docker_assets_exist():
    for name in ["Dockerfile", "docker-compose.yml", ".dockerignore", "Makefile"]:
        assert (ROOT / name).is_file(), name


def test_compose_defines_full_stack():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service in ["qdrant:", "api:", "ui:", "dev:"]:
        assert service in text, service


def test_dockerfile_is_multistage_nonroot_and_caches_models():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM ") >= 3
    assert "USER " in text
    assert "HF_HOME" in text


def test_services_that_call_the_host_llm_map_host_gateway():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    for name in ["api", "dev", "api-gpu"]:
        extra_hosts = " ".join(compose["services"][name].get("extra_hosts", []))

        assert "host.docker.internal:host-gateway" in extra_hosts, name


def test_ui_healthcheck_checks_streamlit_not_the_api_port():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    check = " ".join(compose["services"]["ui"]["healthcheck"]["test"])

    assert "8501" in check
    assert "8000" not in check
