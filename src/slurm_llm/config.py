"""Configuration files and paths for slurm-llm.

Clusters come from rclust's config (`~/.config/rclust/config.yaml`, or `$RCLUST_CONFIG`):
login host, user, Slurm account and GPU types. slurm-llm adds vLLM-specific settings per
cluster in `~/.config/slurm-llm/clusters.yaml` (paths, templates, modules). Models and
resource profiles live next to it; the packaged defaults are used for any file that is missing.

Files follow the XDG Base Directory layout:

- settings (clusters.yaml, model_registry.yaml, ..., .hf_token, templates/): `$XDG_CONFIG_HOME/slurm-llm`
  (default `~/.config/slurm-llm`, or `$SLURM_LLM_CONFIG_DIR`)
- state (the job cache, the tunnel registry): `$XDG_STATE_HOME/slurm-llm`
  (default `~/.local/state/slurm-llm`, or `$SLURM_LLM_STATE_DIR`)

Lock files are kept in the system temporary directory, which is always host-local (flock often
fails on the network file systems that hold home directories on clusters).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent


class SlurmLLMError(Exception):
    """An expected failure with a message meant for the user (no traceback needed)."""


class ConfigError(SlurmLLMError, ValueError):
    """A configuration file is missing or invalid."""


def env(name: str) -> str | None:
    """$SLURM_LLM_<name>; None when it is not set (or empty)."""
    return os.environ.get(f"SLURM_LLM_{name}") or None


APP_NAME = "slurm-llm"


def xdg_dir(variable: str, default: str) -> Path:
    """An XDG base directory: $variable if it is an absolute path (as the spec requires), else ~/default."""
    value = os.environ.get(variable, "")
    return Path(value) if os.path.isabs(value) else Path.home() / default


def user_config_dir() -> Path:
    """Settings: $SLURM_LLM_CONFIG_DIR, else $XDG_CONFIG_HOME/slurm-llm (~/.config/slurm-llm)."""
    if env("CONFIG_DIR"):
        return Path(env("CONFIG_DIR")).expanduser()
    return xdg_dir("XDG_CONFIG_HOME", ".config") / APP_NAME


def state_dir() -> Path:
    """Local state (job cache, tunnel registry): $SLURM_LLM_STATE_DIR, else
    $XDG_STATE_HOME/slurm-llm (~/.local/state/slurm-llm)."""
    if env("STATE_DIR"):
        return Path(env("STATE_DIR")).expanduser()
    return xdg_dir("XDG_STATE_HOME", ".local/state") / APP_NAME


def user_template_dir() -> Path:
    """Your own job templates, which take precedence over the packaged ones: <config dir>/templates."""
    return user_config_dir() / "templates"


def tilde(path: Path | str) -> str:
    """A path for messages, with the home directory shown as ~."""
    home = str(Path.home())
    path = str(path)
    return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path


def rclust_config_path() -> Path:
    """rclust's config file: $RCLUST_CONFIG, else ~/.config/rclust/config.yaml."""
    if os.environ.get("RCLUST_CONFIG"):
        return Path(os.environ["RCLUST_CONFIG"]).expanduser()
    from rclust.config import user_config_path

    return user_config_path()


def find_template(name: str, template_dir: Path | None = None) -> Path:
    """A template from ~/.config/slurm-llm/templates (your copies) or else the packaged ones."""
    candidates = [(template_dir or user_template_dir()) / name, PACKAGE_DIR / "templates" / name]
    for path in candidates:
        if path.is_file():
            return path
    available = sorted(p.name for p in (PACKAGE_DIR / "templates").iterdir() if p.suffix in (".slurm", ".sh"))
    raise ConfigError(
        f"Template '{name}' not found in {candidates[0].parent} or the packaged templates "
        f"(available: {', '.join(available)})"
    )


def read_hf_token() -> str | None:
    """Hugging Face token from $HF_TOKEN or the file written by `sllm config hf-token`."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = user_config_dir() / ".hf_token"
    if token_file.is_file():
        return token_file.read_text().strip() or None
    return None


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from None
    except OSError as e:
        raise ConfigError(f"Could not read {path}: {e.strerror}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


@dataclass
class GPUType:
    """A GPU type available on a cluster."""

    type: str  # name used in configs, e.g. "h100" or "h100_1g"
    memory: int  # GB per GPU
    slurm_spec: str | None = None  # name Slurm expects (e.g. "h100_1g.10gb"); None: request untyped GPUs
    is_full_gpu: bool = True  # False for MIG slices, which cannot do tensor parallelism


@dataclass
class ClusterConfig:
    """A cluster: connection details from rclust plus slurm-llm's settings."""

    name: str
    host: str
    user: str | None = None  # None: whatever ~/.ssh/config says
    account: str | None = None
    remote_root: str | None = None  # job scripts, logs and endpoint files (default ~/slurm-llm)
    model_path: str | None = None  # Hugging Face cache directory for model weights (and the venv)
    slurm_template: str = "vllm.slurm"
    setup_template: str = "setup_venv.sh"
    gpu_types: list[GPUType] = field(default_factory=list)
    gpu_allocation_mode: str = "per-node"  # "per-node" (--gpus-per-node) or "flexible" (--gpus)
    require_single_node: bool = False
    setup_commands: list[str] = field(default_factory=list)  # e.g. ["module load python/3.12 cuda/12.4"]
    sbatch_args: list[str] = field(default_factory=list)  # e.g. ["--partition=gpu"]
    cpus_per_task: int = 4


@dataclass
class ModelConfig:
    """A model in the registry."""

    id: str  # Hugging Face cache name, e.g. "models--Qwen--Qwen3-8B"
    provider: str = "vllm"
    size: str = "small"
    tags: list[str] = field(default_factory=list)
    model_memory_gb: int | None = None  # override: total GPU memory to request
    max_model_len: int | None = None  # context window passed to vLLM
    vllm_extra_args: list[str] = field(default_factory=list)  # appended verbatim to `vllm serve`
    short_name: str | None = None  # the registry name used in commands and requests, e.g. "qwen-3-8b"

    @property
    def name(self) -> str:
        """The short name, or the served name for a model without one."""
        return self.short_name or self.served_name

    @property
    def served_name(self) -> str:
        """The name vLLM serves the model under (e.g. "Qwen3-8B")."""
        return self.id.split("--")[-1]

    @property
    def hf_repo(self) -> str:
        """The Hugging Face repo id (e.g. "Qwen/Qwen3-8B")."""
        repo = self.id[len("models--"):] if self.id.startswith("models--") else self.id
        return repo.replace("--", "/")


@dataclass
class ResourceConfig:
    """Default Slurm resources for a model size."""

    memory: int  # system RAM in GB
    time: str  # Slurm time limit
    estimated_model_memory: int | None = None  # GPU memory guess when a model was never scanned


@dataclass
class ModelSpecs:
    """Model architecture read from config.json by `sllm scan`, for GPU memory estimates."""

    model_id: str
    model_memory_gb: float  # weights
    num_hidden_layers: int
    num_attention_heads: int  # determines valid tensor-parallel sizes
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int
    max_position_embeddings: int
    scanned_at: str | None = None
    scanned_cluster: str | None = None

    def compute_kv_cache_gb(self, max_model_len: int | None = None) -> float:
        """KV cache for one full-length sequence, in GB."""
        context_len = max_model_len or self.max_position_embeddings
        kv_bytes_per_token = 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * self.dtype_bytes
        return context_len * kv_bytes_per_token / 1024**3

    def compute_total_gpu_memory(self, max_model_len: int | None = None, overhead_factor: float = 1.25) -> float:
        """Weights plus KV cache, with 25% headroom for vLLM's memory fraction, CUDA context and activations."""
        return (self.model_memory_gb + self.compute_kv_cache_gb(max_model_len)) * overhead_factor


class ConfigLoader:
    """Loads slurm-llm's configuration.

    `config_dir` defaults to `user_config_dir()`; any of model_registry.yaml, resources.yaml that it
    lacks is read from the packaged defaults.
    """

    def __init__(self, config_dir: Path | str | None = None):
        self.config_dir = Path(config_dir).expanduser() if config_dir else user_config_dir()

    def config_file(self, name: str) -> Path:
        """`name` in the config directory, or the packaged default when it is missing there."""
        path = self.config_dir / name
        if not path.exists() and (PACKAGE_DIR / "configs" / name).exists():
            return PACKAGE_DIR / "configs" / name
        return path

    # --- clusters ------------------------------------------------------------------------------

    def rclust_config_path(self) -> Path:
        return rclust_config_path()

    def overlay_path(self) -> Path | None:
        """clusters.yaml with slurm-llm's per-cluster settings, if it exists."""
        path = self.config_dir / "clusters.yaml"
        return path if path.exists() else None

    def load_clusters(self) -> dict[str, ClusterConfig]:
        """Clusters from rclust's config, merged with slurm-llm's settings for each."""
        path = self.rclust_config_path()
        if not path.exists():
            raise ConfigError(
                f"No cluster config at {tilde(path)}.\n"
                "slurm-llm reads your clusters from rclust's config. Create it with `rclust config` "
                "(rclust is installed with slurm-llm), then run `sllm init`."
            )
        rclust_clusters = _load_yaml(path).get("clusters") or {}
        if not isinstance(rclust_clusters, dict) or not rclust_clusters:
            raise ConfigError(f"{path}: no clusters defined (run `rclust config` to add one)")

        overlay_file = self.overlay_path()
        overlay = (_load_yaml(overlay_file).get("clusters") or {}) if overlay_file else {}

        clusters = {}
        for name, base in rclust_clusters.items():
            base = base or {}
            extra = overlay.get(name) or {}
            if not isinstance(base, dict) or not isinstance(extra, dict):
                raise ConfigError(f"cluster '{name}': expected a mapping of settings")
            clusters[name] = ClusterConfig(
                name=name,
                host=base.get("host") or name,
                user=base.get("user"),
                account=extra.get("account", base.get("account")),
                remote_root=extra.get("remote_root") or "~/slurm-llm",
                model_path=extra.get("model_path"),
                slurm_template=extra.get("slurm_template", "vllm.slurm"),
                setup_template=extra.get("setup_template", "setup_venv.sh"),
                gpu_types=_merge_gpu_types(name, rclust_gpu_types(base), extra.get("gpu_types") or []),
                gpu_allocation_mode=extra.get("gpu_allocation_mode", "per-node"),
                require_single_node=bool(extra.get("require_single_node", base.get("require_single_node", False))),
                setup_commands=_as_list(extra.get("setup_commands")),
                sbatch_args=_as_list(extra.get("sbatch_args")),
                cpus_per_task=int(extra.get("cpus_per_task", 4)),
            )
        return clusters

    def get_cluster(self, name: str) -> ClusterConfig:
        clusters = self.load_clusters()
        if name not in clusters:
            raise ConfigError(f"Unknown cluster '{name}' (configured: {', '.join(clusters)})")
        return clusters[name]

    # --- models and resources ------------------------------------------------------------------

    def load_models(self) -> dict[str, ModelConfig]:
        """Models by short name and by full id."""
        path = self.config_file("model_registry.yaml")
        if not path.exists():
            raise ConfigError(f"Model registry not found at {path} (run `sllm init`)")
        models = {}
        for entry in _load_yaml(path).get("models") or []:
            if not isinstance(entry, dict) or "id" not in entry:
                raise ConfigError(f"{path}: every model needs an 'id'")
            model = ModelConfig(
                id=entry["id"],
                provider=entry.get("provider", "vllm"),
                size=entry.get("size", "small"),
                tags=entry.get("tags") or [],
                model_memory_gb=entry.get("model_memory_gb"),
                max_model_len=entry.get("max_model_len"),
                vllm_extra_args=entry.get("vllm_extra_args") or [],
                short_name=entry.get("short_name"),
            )
            models[model.name] = model
            models[model.id] = model
        return models

    def load_resources(self) -> dict[str, ResourceConfig]:
        path = self.config_file("resources.yaml")
        if not path.exists():
            raise ConfigError(f"Resource profiles not found at {path} (run `sllm init`)")
        resources = {}
        for size, entry in _load_yaml(path).items():
            try:
                resources[size] = ResourceConfig(
                    memory=entry["memory"],
                    time=entry["time"],
                    estimated_model_memory=entry.get("estimated_model_memory"),
                )
            except (KeyError, TypeError):
                raise ConfigError(f"{path}: size '{size}' needs 'memory' and 'time'") from None
        return resources

    def resource_for(self, model: ModelConfig) -> ResourceConfig:
        resources = self.load_resources()
        if model.size not in resources:
            raise ConfigError(
                f"Model size '{model.size}' of {model.id} has no profile in resources.yaml "
                f"(sizes: {', '.join(resources)})"
            )
        return resources[model.size]

    def resolve_model(self, model_name: str) -> ModelConfig | None:
        """A model by short name, id or Hugging Face repo id ("Qwen/Qwen3-8B"); otherwise the first
        whose name contains `model_name`."""
        models = self.load_models()
        if model_name in models:
            return models[model_name]
        if "/" in model_name and "models--" + model_name.replace("/", "--") in models:
            return models["models--" + model_name.replace("/", "--")]
        needle = model_name.lower()
        return next((m for key, m in models.items() if needle in key.lower()), None)

    def require_model(self, model_name: str) -> ModelConfig:
        model = self.resolve_model(model_name)
        if model is None:
            raise ConfigError(
                f"Model '{model_name}' is not in the model registry "
                f"({tilde(self.config_file('model_registry.yaml'))}); see `sllm models`"
            )
        return model

    # --- model specs (written by `sllm scan`) ---------------------------------------------

    def _get_model_specs_path(self) -> Path:
        return self.config_dir / "model_specs.yaml"

    def load_model_specs(self) -> dict[str, ModelSpecs]:
        path = self._get_model_specs_path()
        if not path.exists():
            return {}
        try:
            specs = {}
            for model_id, s in (_load_yaml(path).get("models") or {}).items():
                specs[model_id] = ModelSpecs(
                    model_id=model_id,
                    model_memory_gb=s["model_memory_gb"],
                    num_hidden_layers=s["num_hidden_layers"],
                    num_attention_heads=s.get("num_attention_heads", s["num_key_value_heads"]),
                    num_key_value_heads=s["num_key_value_heads"],
                    head_dim=s["head_dim"],
                    dtype_bytes=s["dtype_bytes"],
                    max_position_embeddings=s["max_position_embeddings"],
                    scanned_at=s.get("scanned_at"),
                    scanned_cluster=s.get("scanned_cluster"),
                )
            return specs
        except (ConfigError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Ignoring unreadable {path}: {e}")
            return {}

    def get_model_specs(self, model_id: str) -> ModelSpecs | None:
        return self.load_model_specs().get(model_id)

    def save_model_specs(self, specs: ModelSpecs) -> None:
        path = self._get_model_specs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _load_yaml(path) if path.exists() else {}
        data.setdefault("_comment", "Written by `sllm scan`; edits are overwritten on re-scan")
        data.setdefault("models", {})[specs.model_id] = {
            key: getattr(specs, key)
            for key in (
                "model_memory_gb", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "dtype_bytes", "max_position_embeddings", "scanned_at", "scanned_cluster",
            )
        }
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def get_total_gpu_memory(self, model_config: ModelConfig) -> float | None:
        """GPU memory the model needs: the registry's model_memory_gb if set, else the scanned estimate."""
        if model_config.model_memory_gb:
            return float(model_config.model_memory_gb)
        specs = self.get_model_specs(model_config.id)
        if specs:
            return specs.compute_total_gpu_memory(model_config.max_model_len)
        return None

    # --- cluster preferences -------------------------------------------------------------------

    def load_cluster_preferences(self) -> dict[str, list[str]]:
        """Model size -> preferred clusters, from cluster_preferences.yaml (optional)."""
        path = self.config_dir / "cluster_preferences.yaml"
        if not path.exists():
            return {}
        return _load_yaml(path).get("cluster_preferences") or {}


def rclust_gpu_types(cluster: dict | None) -> list:
    """GPU types of a cluster entry in rclust's config, which accepts them at the top level of the
    cluster or under `resources:` (the top level wins, as in rclust)."""
    cluster = cluster if isinstance(cluster, dict) else {}
    if "gpu_types" in cluster:
        return cluster.get("gpu_types") or []
    resources = cluster.get("resources")
    return (resources.get("gpu_types") or []) if isinstance(resources, dict) else []


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()]
    return [str(v) for v in value]


def _merge_gpu_types(cluster: str, base: list, extra: list) -> list[GPUType]:
    """GPU types from rclust's config, completed by entries of the same type in clusters.yaml.

    rclust only needs a type's name and Slurm spec; slurm-llm also needs its memory (GB) to size
    jobs, which may be given in either file. Types without a memory are not used for sizing.
    """
    specs: dict[str, dict] = {}
    for g in [*base, *extra]:
        if not isinstance(g, dict) or not g.get("type"):
            raise ConfigError(f"cluster '{cluster}': every GPU type needs a 'type'")
        specs[g["type"]] = {**specs.get(g["type"], {}), **g}
    merged = []
    for spec in specs.values():
        if spec.get("memory") is None:
            logger.debug(f"cluster '{cluster}': GPU type '{spec['type']}' has no memory; not used for sizing")
            continue
        merged.append(
            GPUType(
                type=spec["type"],
                memory=int(spec["memory"]),
                slurm_spec=spec.get("slurm_spec"),
                is_full_gpu=spec.get("is_full_gpu", True),
            )
        )
    return merged


def __getattr__(name):
    if name == "RemoteLLMError":  # old name
        from . import _renamed

        return _renamed(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
