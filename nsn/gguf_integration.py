"""Helpers for loading GGUF models into Hugging Face ``transformers``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - optional dependency
    import gguf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gguf = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore
    LlamaConfig = None  # type: ignore
    LlamaForCausalLM = None  # type: ignore


def _require_dependencies() -> None:
    if gguf is None:  # pragma: no cover - import guard
        raise ImportError("The 'gguf' package is required to load GGUF models")
    if LlamaForCausalLM is None:  # pragma: no cover - import guard
        raise ImportError("transformers is required to load GGUF models")


def _field_value(reader: "gguf.GGUFReader", key: str, *, arch: Optional[str] = None):
    formatted_key = key.format(arch=arch) if arch is not None else key
    field = reader.get_field(formatted_key)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            value = value[0]
    return value


def _require_field(reader: "gguf.GGUFReader", key: str, *, arch: Optional[str] = None):
    value = _field_value(reader, key, arch=arch)
    if value is None:
        suffix = f" for architecture '{arch}'" if arch is not None else ""
        raise ValueError(f"Required GGUF field '{key}'{suffix} is missing")
    return value


def _resolve_architecture(reader: "gguf.GGUFReader") -> Tuple["gguf.MODEL_ARCH", str]:
    arch_value = _field_value(reader, gguf.KEY_GENERAL_ARCHITECTURE)
    if arch_value is None:
        raise ValueError("GGUF file does not specify an architecture")
    for arch_enum, name in gguf.MODEL_ARCH_NAMES.items():
        if name == arch_value:
            return arch_enum, name
    raise ValueError(f"Unsupported GGUF architecture: {arch_value}")


def _rope_scaling(reader: "gguf.GGUFReader", arch_name: str) -> Optional[Dict[str, float]]:
    scaling_type = _field_value(reader, gguf.KEY_ROPE_SCALING_TYPE, arch=arch_name)
    if scaling_type is None:
        return None
    factor = _field_value(reader, gguf.KEY_ROPE_SCALING_FACTOR, arch=arch_name)
    if factor is None:
        return None
    scaling: Dict[str, float] = {"type": scaling_type, "factor": float(factor)}
    original_ctx = _field_value(reader, gguf.KEY_ROPE_SCALING_ORIG_CTX_LEN, arch=arch_name)
    if original_ctx is not None:
        scaling["original_max_position_embeddings"] = int(original_ctx)
    return scaling


def load_gguf_causal_lm(
    path: str | Path,
    *,
    tokenizer_path: Optional[str | Path] = None,
    dtype: torch.dtype = torch.float32,
):
    """Load a GGUF causal language model into a ``transformers`` model."""

    _require_dependencies()
    reader = gguf.GGUFReader(str(path))
    arch_enum, arch_name = _resolve_architecture(reader)
    if arch_enum not in (gguf.MODEL_ARCH.LLAMA, gguf.MODEL_ARCH.LLAMA4):
        raise ValueError(
            f"Only LLaMA-family GGUF models are supported, found architecture '{arch_name}'"
        )

    n_layers = int(_require_field(reader, gguf.KEY_BLOCK_COUNT, arch=arch_name))
    hidden_size = int(_require_field(reader, gguf.KEY_EMBEDDING_LENGTH, arch=arch_name))
    intermediate_size = int(
        _require_field(reader, gguf.KEY_FEED_FORWARD_LENGTH, arch=arch_name)
    )
    n_heads = int(_require_field(reader, gguf.KEY_ATTENTION_HEAD_COUNT, arch=arch_name))
    n_kv_heads_value = _field_value(reader, gguf.KEY_ATTENTION_HEAD_COUNT_KV, arch=arch_name)
    n_kv_heads = int(n_kv_heads_value) if n_kv_heads_value is not None else n_heads
    context_length = int(_require_field(reader, gguf.KEY_CONTEXT_LENGTH, arch=arch_name))
    rope_theta_value = _field_value(reader, gguf.KEY_ROPE_FREQ_BASE, arch=arch_name)
    rope_theta = float(rope_theta_value) if rope_theta_value is not None else 10000.0
    eps_value = _field_value(reader, gguf.KEY_ATTENTION_LAYERNORM_RMS_EPS, arch=arch_name)
    rms_norm_eps = float(eps_value) if eps_value is not None else 1e-5
    vocab_size = int(_require_field(reader, gguf.KEY_VOCAB_SIZE))

    config_kwargs = dict(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=context_length,
        rms_norm_eps=rms_norm_eps,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
    )
    rope_scaling = _rope_scaling(reader, arch_name)
    if rope_scaling is not None:
        config_kwargs["rope_scaling"] = rope_scaling

    config = LlamaConfig(**config_kwargs)  # type: ignore[call-arg]

    tensor_map = gguf.get_tensor_name_map(arch_enum, n_layers)
    state_dict: Dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        mapped_name = tensor_map.get_name(tensor.name)
        if mapped_name is None:
            continue
        np_tensor = gguf.dequantize(tensor.data, tensor.tensor_type)
        tensor_value = torch.from_numpy(np_tensor.copy()).to(dtype)
        state_dict[mapped_name] = tensor_value

    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(f"Missing tensors when loading GGUF model: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected tensors when loading GGUF model: {sorted(unexpected)}")

    bos_id = _field_value(reader, gguf.KEY_TOKENIZER_BOS_ID)
    eos_id = _field_value(reader, gguf.KEY_TOKENIZER_EOS_ID)
    pad_id = _field_value(reader, gguf.KEY_TOKENIZER_PAD_ID)
    unk_id = _field_value(reader, gguf.KEY_TOKENIZER_UNK_ID)
    if bos_id is not None:
        model.config.bos_token_id = int(bos_id)
    if eos_id is not None:
        model.config.eos_token_id = int(eos_id)
    if pad_id is not None:
        model.config.pad_token_id = int(pad_id)
    if unk_id is not None:
        model.config.unk_token_id = int(unk_id)

    tokenizer = None
    if tokenizer_path is not None:
        if AutoTokenizer is None:  # pragma: no cover - import guard
            raise ImportError("transformers is required to load tokenizers")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model.eval()
    return model, tokenizer


__all__ = ["load_gguf_causal_lm"]

