"""Model scanning utilities for extracting specs from HuggingFace models.

Scans downloaded models via SSH to extract architecture parameters from
config.json and compute model memory from file sizes.
"""

import json
import logging
from datetime import datetime

from rich.console import Console

from .config import ModelSpecs, ClusterConfig
from .job import model_snapshot_expr
from .ssh import SSHClient, shell_path

logger = logging.getLogger(__name__)
console = Console()


# Mapping of torch dtype strings to byte sizes
DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    # Common variations
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
}


def scan_model_specs(
    ssh_client: SSHClient,
    model_id: str,
    cluster_config: ClusterConfig,
) -> ModelSpecs | None:
    """
    Scan a downloaded model to extract architecture specs.
    
    Args:
        ssh_client: SSH client for the cluster
        model_id: Model ID (e.g., "models--Qwen--Qwen2.5-7B-Instruct")
        cluster_config: Cluster configuration with model_path
    
    Returns:
        ModelSpecs if successful, None if scan failed
    """
    model_path = cluster_config.model_path
    if not model_path:
        logger.warning(f"No model_path configured for cluster {cluster_config.name}")
        return None

    model_dir = f"{model_path}/{model_id}"

    # Step 1: Read config.json
    config_data = _read_model_config(ssh_client, model_path, model_id)
    if not config_data:
        logger.warning(f"Could not read config.json for {model_id}")
        return None

    # Step 2: Compute model memory from architecture (not disk size)
    model_memory_gb = _compute_model_memory_from_config(config_data)
    if model_memory_gb is None:
        # Fallback to disk size if we can't compute from architecture
        model_memory_gb = _get_model_file_size(ssh_client, model_dir)
        if model_memory_gb is None:
            logger.warning(f"Could not determine model size for {model_id}")
            return None
        logger.debug(f"Using disk-based memory estimate for {model_id}: {model_memory_gb:.1f}GB")
    else:
        logger.debug(f"Computed memory from architecture for {model_id}: {model_memory_gb:.1f}GB")

    # Step 3: Extract architecture parameters
    try:
        specs = _parse_config_to_specs(
            model_id=model_id,
            config_data=config_data,
            model_memory_gb=model_memory_gb,
            cluster_name=cluster_config.name,
        )
        return specs
    except Exception as e:
        logger.warning(f"Could not parse config for {model_id}: {e}")
        return None


def _detect_quantization(
    config_data: dict, default_bytes: int
) -> tuple[float, bool, str | None]:
    """Read ``quantization_config`` from a HF config.json.

    Returns ``(bytes_per_quantized_param, attention_quantized, quant_method)``.

    Quantization in HF configs is described by a ``quantization_config`` dict
    with a ``quant_method`` field. For models that quantize only a subset of
    weights (e.g. gpt-oss mxfp4 keeps attention in bf16), the subset is
    listed in ``modules_to_not_convert``. We use that list to decide whether
    attention is quantized — embeddings/LM head are always treated as native
    dtype since they're almost universally excluded from quantization.

    Quant byte estimates include typical scale/zero overhead:
    - 4-bit families (mxfp4, nf4, gptq, awq, bnb_4bit): 0.53 bytes/param
    - 8-bit families (fp8, int8, bnb_8bit):              1.06 bytes/param
    """
    qcfg = config_data.get("quantization_config")
    if not isinstance(qcfg, dict):
        return float(default_bytes), True, None

    method = (qcfg.get("quant_method") or "").lower()
    if any(t in method for t in ("4bit", "fp4", "nf4", "gptq", "awq")):
        quant_bytes = 0.53
    elif any(t in method for t in ("8bit", "fp8", "int8")):
        quant_bytes = 1.06
    else:
        return float(default_bytes), True, method

    skip_modules = qcfg.get("modules_to_not_convert") or []
    skip_str = " ".join(skip_modules) if isinstance(skip_modules, list) else str(skip_modules)
    attn_keywords = ("self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")
    attention_quantized = not any(kw in skip_str for kw in attn_keywords)

    return quant_bytes, attention_quantized, method


def _detect_moe(text_cfg: dict) -> tuple[int, int, int | None]:
    """Detect MoE structure from a transformer config.

    Returns ``(num_routed_experts, num_shared_experts, moe_intermediate_size)``.
    ``num_routed_experts`` is 0 for dense models. ``moe_intermediate_size`` is
    None when the per-expert FFN width matches the dense ``intermediate_size``
    (caller should fall back to it in that case).

    Detection covers the major MoE families: Mixtral / gpt-oss / Gemma MoE
    (``num_local_experts``), generic Qwen-MoE (``num_experts``), and
    DeepSeek-style (``n_routed_experts`` + ``n_shared_experts``).
    """
    num_routed = (
        text_cfg.get("num_local_experts")
        or text_cfg.get("num_routed_experts")
        or text_cfg.get("n_routed_experts")
        or text_cfg.get("num_experts")
        or 0
    )
    num_shared = (
        text_cfg.get("n_shared_experts")
        or text_cfg.get("num_shared_experts")
        or 0
    )
    moe_int = (
        text_cfg.get("moe_intermediate_size")
        or text_cfg.get("expert_intermediate_size")
    )
    return int(num_routed), int(num_shared), moe_int


def _compute_model_memory_from_config(config_data: dict) -> float | None:
    """
    Compute model memory from architecture parameters in config.json.

    This is more accurate than disk size because:
    - Disk size includes metadata overhead
    - Disk block alignment can inflate reported size
    - Some models have non-weight files in blobs

    For MoE models, total weight memory is dominated by the routed expert
    bank: ``num_routed_experts × per-expert FFN params``. The scanner
    detects MoE from the config and accounts for all experts (routed +
    shared), since vLLM loads the full bank into VRAM even though only a
    subset is active per token.

    Args:
        config_data: Parsed config.json

    Returns:
        Estimated model memory in GB, or None if can't compute
    """
    # Handle nested configs (e.g., multimodal models with text_config)
    if "text_config" in config_data:
        text_cfg = config_data["text_config"]
    else:
        text_cfg = config_data

    # Get data type bytes
    torch_dtype = config_data.get("torch_dtype", "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype, 2)

    # Try to get vocab_size and hidden_size (required for embedding layer)
    vocab_size = text_cfg.get("vocab_size")
    hidden_size = text_cfg.get("hidden_size", text_cfg.get("n_embd", text_cfg.get("d_model")))
    num_layers = text_cfg.get("num_hidden_layers", text_cfg.get("n_layer", text_cfg.get("num_layers")))
    intermediate_size = text_cfg.get("intermediate_size", text_cfg.get("n_inner"))

    if not all([vocab_size, hidden_size, num_layers]):
        logger.debug("Missing required fields for architecture-based memory computation")
        return None

    # Default intermediate_size to 4 * hidden_size (common for transformer FFN)
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    # Get attention head info
    num_attention_heads = text_cfg.get(
        "num_attention_heads",
        text_cfg.get("n_head", text_cfg.get("num_heads", hidden_size // 64))
    )
    num_kv_heads = text_cfg.get(
        "num_key_value_heads",
        text_cfg.get("num_kv_heads", num_attention_heads)
    )

    # Compute parameter count (approximate)
    # Embedding: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size

    # Each transformer layer:
    # - Attention QKV projection: hidden_size * (hidden_size + 2 * kv_dim)
    # - Attention output: hidden_size * hidden_size
    # - FFN: hidden_size * intermediate_size * 2 (up + down, plus gate for SwiGLU)
    # - LayerNorms: negligible

    head_dim = hidden_size // num_attention_heads
    kv_dim = num_kv_heads * head_dim

    # Q projection
    q_params = hidden_size * hidden_size
    # KV projection (may be smaller with GQA)
    kv_params = hidden_size * kv_dim * 2
    # Output projection
    o_params = hidden_size * hidden_size

    attention_params = q_params + kv_params + o_params

    # Check for SwiGLU (used by Llama, Qwen, etc.) - has gate projection
    has_gate = text_cfg.get("hidden_act") in ["silu", "swiglu"] or "gate_proj" in str(config_data)
    gate_mult = 3 if has_gate else 2

    # MoE: replicate FFN across all loaded experts.
    num_routed, num_shared, moe_int = _detect_moe(text_cfg)
    is_moe = num_routed > 0
    if is_moe:
        per_expert_int = moe_int if moe_int else intermediate_size
        # Total experts whose weights must reside in VRAM.
        total_experts = num_routed + num_shared
        ffn_params = total_experts * hidden_size * per_expert_int * gate_mult
        # Router gating: small (num_routed × hidden_size per layer). Negligible
        # vs expert bank, but include for completeness.
        router_params = num_routed * hidden_size
    else:
        ffn_params = hidden_size * intermediate_size * gate_mult
        router_params = 0

    # Quantization: most quantized checkpoints leave embeddings/LM head/router
    # at native dtype and apply low-bit storage only to linear weights
    # (attention and/or FFN). gpt-oss mxfp4 quantizes FFN experts but keeps
    # attention in bf16; many dense 4-bit quants do both.
    quant_bytes, attn_quantized, quant_method = _detect_quantization(
        config_data, dtype_bytes
    )

    attention_bytes = quant_bytes if (quant_method and attn_quantized) else dtype_bytes
    ffn_bytes = quant_bytes if quant_method else dtype_bytes

    # LM head (often tied to embeddings in modern models, but count it anyway for safety)
    lm_head_params = vocab_size * hidden_size

    # Per-bucket memory (bytes per param differs across buckets when quantized).
    layer_attn_bytes = num_layers * attention_params * attention_bytes
    layer_ffn_bytes = num_layers * ffn_params * ffn_bytes
    layer_router_bytes = num_layers * router_params * dtype_bytes
    embed_bytes = (embedding_params + lm_head_params) * dtype_bytes

    total_bytes = embed_bytes + layer_attn_bytes + layer_ffn_bytes + layer_router_bytes
    memory_gb = total_bytes / (1024**3)
    total_params = (
        embedding_params
        + num_layers * (attention_params + ffn_params + router_params)
        + lm_head_params
    )

    extras = []
    if is_moe:
        extras.append(
            f"MoE: {num_routed}+{num_shared} experts, "
            f"int={moe_int or intermediate_size}"
        )
    if quant_method:
        extras.append(
            f"quant={quant_method} "
            f"({quant_bytes}B/param, attn_quant={attn_quantized})"
        )
    extras_str = ", ".join(extras)

    logger.debug(
        f"Architecture-based estimate: {total_params/1e9:.1f}B params, "
        f"{memory_gb:.1f}GB ({torch_dtype}"
        + (f", {extras_str})" if extras_str else ")")
    )

    return round(memory_gb, 2)


def _read_model_config(ssh_client: SSHClient, model_path: str, model_id: str) -> dict | None:
    """config.json of the model's current snapshot in the Hugging Face cache, or None."""
    snapshot = model_snapshot_expr(model_path, model_id)
    exit_code, output, stderr = ssh_client.execute_command(f'cat "{snapshot}/config.json"', timeout=30)
    if exit_code != 0 or not output.strip():
        logger.debug(f"No config.json for {model_id} under {model_path}: {stderr.strip()}")
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in config.json of {model_id}: {e}")
        return None


def _get_model_file_size(ssh_client: SSHClient, model_dir: str) -> float | None:
    """Size of the model's files in the Hugging Face cache (its blobs/ directory), in GB."""
    exit_code, output, _ = ssh_client.execute_command(f"du -sb {shell_path(model_dir + '/blobs')} | cut -f1", timeout=60)
    try:
        return int(output.strip()) / 1024**3 if exit_code == 0 else None
    except ValueError:
        return None


def _parse_config_to_specs(
    model_id: str,
    config_data: dict,
    model_memory_gb: float,
    cluster_name: str,
) -> ModelSpecs:
    """
    Parse HuggingFace config.json into ModelSpecs.
    
    Args:
        model_id: Model identifier
        config_data: Parsed config.json
        model_memory_gb: Model file size in GB
        cluster_name: Cluster where model was scanned
    
    Returns:
        ModelSpecs instance
    
    Raises:
        KeyError: If required fields are missing
    """
    # Handle nested configs (e.g., Gemma3 multimodal models with text_config)
    # Prefer text_config if present, otherwise use root config
    if "text_config" in config_data:
        text_cfg = config_data["text_config"]
    else:
        text_cfg = config_data
    
    # Extract architecture parameters
    # Different models may use different key names
    
    # Number of layers
    num_hidden_layers = text_cfg.get(
        "num_hidden_layers",
        text_cfg.get("n_layer", text_cfg.get("num_layers"))
    )
    if num_hidden_layers is None:
        raise KeyError("Could not find num_hidden_layers in config")

    # Hidden size
    hidden_size = text_cfg.get(
        "hidden_size",
        text_cfg.get("n_embd", text_cfg.get("d_model"))
    )
    if hidden_size is None:
        raise KeyError("Could not find hidden_size in config")

    # Number of attention heads
    num_attention_heads = text_cfg.get(
        "num_attention_heads",
        text_cfg.get("n_head", text_cfg.get("num_heads"))
    )
    
    # Number of KV heads (for GQA models)
    # Falls back to num_attention_heads if not using GQA
    num_key_value_heads = text_cfg.get(
        "num_key_value_heads",
        text_cfg.get("num_kv_heads", num_attention_heads)
    )

    # Head dimension
    head_dim = text_cfg.get("head_dim")
    
    # If num_attention_heads is missing, try to infer from head_dim and hidden_size
    # or use reasonable defaults based on model size
    if num_attention_heads is None:
        if head_dim and hidden_size:
            num_attention_heads = hidden_size // head_dim
        else:
            # Fallback: estimate based on hidden_size (common patterns)
            # Most models use head_dim=64 or 128
            num_attention_heads = hidden_size // 128 if hidden_size >= 2048 else hidden_size // 64
            logger.debug(f"Estimated num_attention_heads={num_attention_heads} from hidden_size={hidden_size}")
    
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    # Max position embeddings (context length)
    # For models with sliding window / hybrid attention (e.g. Gemma 3),
    # text_config.max_position_embeddings may be the local window size,
    # not the true max context.  Prefer the root-level value when it's
    # larger, as it typically represents the full supported context.
    text_max = text_cfg.get(
        "max_position_embeddings",
        text_cfg.get("max_seq_len"),
    )
    root_max = config_data.get(
        "max_position_embeddings",
        config_data.get("max_seq_len", config_data.get("n_positions")),
    )

    has_sliding_window = (
        text_cfg.get("sliding_window") is not None
        or text_cfg.get("sliding_window_size") is not None
        or config_data.get("sliding_window") is not None
    )

    if text_max and root_max and root_max > text_max:
        max_position_embeddings = root_max
    elif text_max is not None:
        max_position_embeddings = text_max
    elif root_max is not None:
        max_position_embeddings = root_max
    else:
        max_position_embeddings = 4096

    if has_sliding_window and max_position_embeddings <= 4096:
        logger.warning(
            f"Model {model_id} has sliding_window attention with "
            f"max_position_embeddings={max_position_embeddings}. "
            f"This may be the local window size, not the true max context. "
            f"Consider setting max_model_len explicitly in the registry."
        )

    # Data type
    torch_dtype = config_data.get("torch_dtype", "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype, 2)  # Default to 2 (bf16)

    return ModelSpecs(
        model_id=model_id,
        model_memory_gb=round(model_memory_gb, 2),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        max_position_embeddings=max_position_embeddings,
        scanned_at=datetime.now().isoformat(),
        scanned_cluster=cluster_name,
    )


def print_memory_estimates(specs: ModelSpecs, model_name: str) -> None:
    """
    Print GPU memory estimates for different context lengths.
    
    Args:
        specs: Model specifications
        model_name: Display name for the model
    """
    console.print(f"\n[bold]GPU Memory Estimates for {model_name}:[/bold]")
    console.print(f"  Model weights: {specs.model_memory_gb:.1f} GB")
    console.print(
        f"  Architecture: {specs.num_hidden_layers} layers, "
        f"{specs.num_key_value_heads} KV heads, "
        f"head_dim={specs.head_dim}, "
        f"{'bf16' if specs.dtype_bytes == 2 else 'fp32'}"
    )
    console.print(f"  Native context: {specs.max_position_embeddings:,} tokens")
    console.print()

    # Show estimates for common context lengths
    context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    
    # Filter to only show lengths <= native max
    context_lengths = [c for c in context_lengths if c <= specs.max_position_embeddings]
    
    # Always include native max if not already there
    if specs.max_position_embeddings not in context_lengths:
        context_lengths.append(specs.max_position_embeddings)
    
    console.print("  [dim]Context  │  KV Cache  │  Total GPU Memory[/dim]")
    console.print("  [dim]─────────┼────────────┼──────────────────[/dim]")
    
    for ctx_len in sorted(context_lengths):
        kv_cache = specs.compute_kv_cache_gb(ctx_len)
        total = specs.compute_total_gpu_memory(ctx_len)
        
        # Format context length nicely
        if ctx_len >= 1000:
            ctx_str = f"{ctx_len // 1024}K".rjust(6)
        else:
            ctx_str = str(ctx_len).rjust(6)
        
        console.print(
            f"  {ctx_str}   │  {kv_cache:5.1f} GB   │  {total:5.1f} GB"
        )
    
    console.print()
    console.print(
        "[dim]Tip: Set max_model_len in model_registry.yaml to reduce GPU requirements[/dim]"
    )
