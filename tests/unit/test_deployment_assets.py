from __future__ import annotations

from pathlib import Path


def test_pyproject_exposes_dubber_console_script() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    assert 'dubber = "cli:main"' in pyproject
    assert "[build-system]" in pyproject


def test_dockerfile_installs_ffmpeg_and_runs_web_command() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "apt-get" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "pip install" in dockerfile
    assert "dubber" in dockerfile


def test_docker_compose_exposes_web_monitor_and_workspace_volume() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "8080:8080" in compose
    assert "VIDEO_DUBBER_WORKSPACE" in compose
    assert "./workspace:/app/workspace" in compose


def test_ci_workflow_runs_pytest() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pytest -q" in workflow
    assert "ffmpeg" in workflow
    assert "python-version" in workflow
