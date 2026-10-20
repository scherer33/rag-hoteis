from pathlib import Path

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
