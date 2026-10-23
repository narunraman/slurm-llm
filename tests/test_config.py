"""Loading and merging configuration."""

import pytest

from slurm_llm.config import ConfigError, ConfigLoader, ModelSpecs, find_template, read_hf_token


def test_clusters_merge_rclust_and_slurm_llm_settings(loader):
    clusters = loader.load_clusters()
    assert list(clusters) == ["cluster-a", "cluster-b"]

    a = clusters["cluster-a"]
    assert (a.host, a.user, a.account) == ("hpc1.example.org", "alice", "proj-a")
    assert a.model_path == "$SCRATCH/models"
    assert a.setup_commands == ["module load python/3.12"]
    assert a.sbatch_args == ["--partition=gpu"]
    # slurm_spec from rclust, memory from clusters.yaml
    assert [(g.type, g.memory, g.slurm_spec, g.is_full_gpu) for g in a.gpu_types] == [
        ("a100", 80, "a100", True),
        ("a100_3g", 40, "a100_3g.40gb", False),
    ]

    b = clusters["cluster-b"]
    assert b.user is None and b.account is None
    assert b.remote_root == "~/slurm-llm"  # default
    assert b.gpu_allocation_mode == "flexible"
    assert [(g.type, g.slurm_spec) for g in b.gpu_types] == [("l40s", "l40s")]  # defined only in clusters.yaml


def test_host_defaults_to_cluster_name_and_gpu_types_without_memory_are_skipped(isolated_home, loader):
    (isolated_home / ".config/rclust/config.yaml").write_text("clusters:\n  myalias:\n    gpu_types: [{type: h100}]\n")
    (isolated_home / ".config/slurm-llm/clusters.yaml").unlink()
    cluster = loader.load_clusters()["myalias"]
    assert cluster.host == "myalias"
    assert cluster.gpu_types == []
    assert cluster.model_path is None


def test_only_clusters_yaml_is_read(isolated_home, loader):
    """A file named cluster.yaml (singular) is ignored."""
    overlay = isolated_home / ".config/slurm-llm"
    (overlay / "clusters.yaml").rename(overlay / "cluster.yaml")
    assert loader.overlay_path() is None
    assert loader.load_clusters()["cluster-a"].model_path is None


def test_xdg_locations(isolated_home, tmp_path, monkeypatch):
    from slurm_llm.config import state_dir, user_config_dir, user_template_dir
    from slurm_llm.job import JobManager

    assert user_config_dir() == isolated_home / ".config/slurm-llm"
    assert state_dir() == isolated_home / ".local/state/slurm-llm"
    assert user_template_dir() == isolated_home / ".config/slurm-llm/templates"
    assert JobManager().job_store_path == isolated_home / ".local/state/slurm-llm/jobs.json"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert user_config_dir() == tmp_path / "cfg/slurm-llm"
    assert state_dir() == tmp_path / "state/slurm-llm"
    monkeypatch.setenv("XDG_STATE_HOME", "relative/path")  # ignored, as the XDG spec requires
    assert state_dir() == isolated_home / ".local/state/slurm-llm"
    monkeypatch.setenv("SLURM_LLM_STATE_DIR", str(tmp_path / "new"))
    assert state_dir() == tmp_path / "new"


def test_template_placeholders_match_what_jobs_provide(loader, tmp_path):
    """Every placeholder the templates README lists is filled in by the job."""
    from slurm_llm.config import ResourceConfig
    from slurm_llm.job import SLURM_TEMPLATE_PLACEHOLDERS, JobManager

    (tmp_path / "t.slurm").write_text("".join(f"{{{p}}}" for p in SLURM_TEMPLATE_PLACEHOLDERS))
    JobManager(template_dir=tmp_path).generate_slurm_script(
        loader.require_model("qwen-3-8b"), ResourceConfig(memory=16, time="1:00:00"),
        loader.get_cluster("cluster-a"), template_name="t.slurm", output_dir="/abs",
    )


def test_missing_rclust_config_explains_what_to_do(isolated_home, loader):
    (isolated_home / ".config/rclust/config.yaml").unlink()
    with pytest.raises(ConfigError, match="rclust config"):
        loader.load_clusters()


def test_rclust_config_env_var_is_honoured(isolated_home, tmp_path, monkeypatch, loader):
    other = tmp_path / "other.yaml"
    other.write_text("clusters:\n  elsewhere:\n    host: h.example.org\n")
    monkeypatch.setenv("RCLUST_CONFIG", str(other))
    assert list(loader.load_clusters()) == ["elsewhere"]


def test_invalid_yaml_is_a_config_error(isolated_home, loader):
    (isolated_home / ".config/slurm-llm/clusters.yaml").write_text("clusters: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        loader.load_clusters()


def test_unknown_cluster(loader):
    with pytest.raises(ConfigError, match="Unknown cluster 'nope'"):
        loader.get_cluster("nope")


def test_packaged_registry_and_resources_are_used_when_missing(loader):
    assert not (loader.config_dir / "model_registry.yaml").exists()
    model = loader.resolve_model("qwen-3-8b")
    assert model.id == "models--Qwen--Qwen3-8B"
    assert model.served_name == "Qwen3-8B"
    assert model.hf_repo == "Qwen/Qwen3-8B"
    assert loader.resource_for(model).time


def test_model_resolution(isolated_home, loader):
    (loader.config_dir / "model_registry.yaml").write_text(
        "models:\n"
        "  - id: models--org--Alpha-7B\n    short_name: alpha\n    size: small\n    max_model_len: 4096\n"
        "  - id: models--org--Beta-70B\n    size: enormous\n    model_memory_gb: 160\n"
    )
    assert loader.resolve_model("alpha").max_model_len == 4096
    assert loader.resolve_model("models--org--Alpha-7B").max_model_len == 4096
    assert loader.resolve_model("Beta-70B").id == "models--org--Beta-70B"  # short name defaults to served name
    assert loader.resolve_model("beta").id == "models--org--Beta-70B"  # partial, case-insensitive
    assert loader.resolve_model("gamma") is None
    with pytest.raises(ConfigError, match="not in the model registry"):
        loader.require_model("gamma")
    with pytest.raises(ConfigError, match="size 'enormous'"):
        loader.resource_for(loader.require_model("beta"))


def test_gpu_memory_prefers_registry_override_then_scanned_specs(loader):
    model = loader.require_model("qwen-3-8b")
    assert loader.get_total_gpu_memory(model) is None

    specs = ModelSpecs(
        model_id=model.id, model_memory_gb=16.0, num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, dtype_bytes=2, max_position_embeddings=32768,
    )
    loader.save_model_specs(specs)
    assert loader.get_model_specs(model.id) == specs
    expected = (16.0 + 32768 * 2 * 36 * 8 * 128 * 2 / 1024**3) * 1.25
    assert loader.get_total_gpu_memory(model) == pytest.approx(expected)

    model.model_memory_gb = 60
    assert loader.get_total_gpu_memory(model) == 60


def test_user_templates_take_precedence(isolated_home):
    assert find_template("vllm.slurm").parent.name == "templates"
    custom = isolated_home / ".config" / "slurm-llm" / "templates"
    custom.mkdir(parents=True)
    (custom / "vllm.slurm").write_text("custom")
    assert find_template("vllm.slurm") == custom / "vllm.slurm"
    with pytest.raises(ConfigError, match="available: setup_venv.sh, vllm.slurm"):
        find_template("missing.slurm")


def test_hf_token_from_environment_or_file(isolated_home, monkeypatch):
    assert read_hf_token() is None
    (isolated_home / ".config/slurm-llm/.hf_token").write_text("hf_fromfile\n")
    assert read_hf_token() == "hf_fromfile"
    monkeypatch.setenv("HF_TOKEN", "hf_fromenv")
    assert read_hf_token() == "hf_fromenv"


def test_config_dir_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_LLM_CONFIG_DIR", str(tmp_path / "cfg"))
    assert ConfigLoader().config_dir == tmp_path / "cfg"


def test_gpu_types_under_resources_in_rclusts_config(isolated_home, loader):
    """rclust accepts gpu_types at the top level of a cluster or under `resources:`."""
    (isolated_home / ".config/rclust/config.yaml").write_text(
        "clusters:\n"
        "  cluster-a:\n"
        "    host: hpc1.example.org\n"
        "    resources:\n"
        "      gpu_types:\n"
        "        - {type: a100, slurm_spec: a100}\n"
        "        - {type: a100_3g, slurm_spec: a100_3g.40gb}\n"
        "  cluster-b:\n"
        "    gpu_types: [{type: h200, slurm_spec: h200, memory: 141}]\n"
        "    resources: {gpu_types: [{type: h100, slurm_spec: h100, memory: 80}]}\n"
    )
    clusters = loader.load_clusters()
    assert [(g.type, g.slurm_spec, g.memory) for g in clusters["cluster-a"].gpu_types] == [
        ("a100", "a100", 80), ("a100_3g", "a100_3g.40gb", 40),
    ]
    assert [g.type for g in clusters["cluster-b"].gpu_types] == ["h200", "l40s"]  # the top level wins


@pytest.mark.parametrize("module", ["slurm_llm", "slurm_llm.client", "slurm_llm.config"])
def test_old_names_still_work_with_a_warning(module):
    import importlib

    import slurm_llm

    mod = importlib.import_module(module)
    old, new = ("RemoteLLMError", slurm_llm.SlurmLLMError) if module.endswith("config") else ("RemoteLLM", slurm_llm.SlurmLLM)
    with pytest.warns(DeprecationWarning, match=f"{old} is now called"):
        assert getattr(mod, old) is new
    if module == "slurm_llm":
        with pytest.warns(DeprecationWarning):
            assert slurm_llm.RemoteLLMError is slurm_llm.SlurmLLMError


CLUSTERS_WITH_COMMENTS = """\
# my clusters
clusters:
  cluster-a:
    remote_root: ~/llm  # where logs go
    setup_commands:
      - module load python/3.12   # same Python as setup
    gpu_types:
      - type: a100
        memory: 80
  cluster-b:
    model_path: /data/models
"""


@pytest.mark.parametrize("cluster, setup_commands", [
    ("cluster-a", ["module load python/3.12", "module load gcc opencv"]),
    ("cluster-b", ["module load gcc opencv"]),  # no setup_commands yet
])
def test_add_setup_command_keeps_comments(isolated_home, cluster, setup_commands):
    loader = ConfigLoader()
    path = loader.config_dir / "clusters.yaml"
    path.write_text(CLUSTERS_WITH_COMMENTS)
    assert loader.add_setup_command(cluster, "module load gcc opencv")
    text = path.read_text()
    assert "# my clusters" in text and "# where logs go" in text and "# same Python as setup" in text
    assert loader.get_cluster(cluster).setup_commands == setup_commands
    assert loader.get_cluster("cluster-a").gpu_types[0].memory == 80


def test_add_setup_command_leaves_unusual_files_alone(isolated_home):
    loader = ConfigLoader()
    path = loader.config_dir / "clusters.yaml"
    path.write_text("clusters:\n  cluster-a:\n    setup_commands: [module load python/3.12]\n")
    assert not loader.add_setup_command("cluster-a", "module load gcc opencv")
    assert path.read_text() == "clusters:\n  cluster-a:\n    setup_commands: [module load python/3.12]\n"
    assert not loader.add_setup_command("no-such-cluster", "x")
