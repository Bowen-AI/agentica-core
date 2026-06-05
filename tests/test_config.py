"""cluster.yaml / plan.yaml loading + defaulting."""

from pathlib import Path

from agentica_core.config import ClusterConfig, PlanConfig

EX = Path(__file__).resolve().parent.parent / "examples"


def test_load_cluster():
    cluster = ClusterConfig.load(EX / "cluster.yaml")
    assert cluster.ssh.host == "login.cluster.edu"
    assert cluster.ssh.user == "me"
    assert cluster.model.name == "qwen3.6"
    assert cluster.model.engine == "ollama"
    assert "a100" in cluster.resources.gpu_types
    assert cluster.gateway.port == 8765


def test_load_plans_and_defaulting():
    cluster = ClusterConfig.load(EX / "cluster.yaml")

    code = PlanConfig.load(EX / "plans" / "code_experiment.yaml")
    assert code.success_criteria.tests
    assert "src/client.py" in code.success_criteria.artifacts
    # model omitted in plan -> defaults to cluster.model
    assert code.effective_model(cluster).name == cluster.model.name
    # resources omitted -> default to cluster slurm
    assert code.effective_resources(cluster).gpu_count == cluster.slurm.gpu_count

    video = PlanConfig.load(EX / "plans" / "video_from_image.yaml")
    assert video.kind == "video"
    assert "output/clip.mp4" in video.success_criteria.artifacts
    assert video.success_criteria.tests is None


def test_plan_requires_goal():
    import pytest

    from agentica_core.config import ConfigError, PlanConfig

    with pytest.raises(ConfigError):
        PlanConfig.from_dict({"title": "x"})
