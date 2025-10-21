"""Hugging Face integration helpers."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

try:  # transformers is optional for the unit tests
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoModelForSequenceClassification = None  # type: ignore
    AutoModelForTokenClassification = AutoModelForQuestionAnswering = None  # type: ignore
    AutoTokenizer = PreTrainedModel = PreTrainedTokenizerBase = None  # type: ignore

from .flops import estimate_model_flops, select_rank_for_target_flops
from .layers import NSNLinear
from .utils import get_rank, set_rank
from .wrap import replace_linear_with_nsn

TaskType = str

_MODULE_PRESETS: Dict[str, Callable[[str, nn.Module], bool]] = {}


def _register_selector(name: str) -> Callable[[Callable[[str, nn.Module], bool]], None]:
    def decorator(fn: Callable[[str, nn.Module], bool]) -> None:
        _MODULE_PRESETS[name] = fn
    return decorator


@_register_selector("mlp")
def _mlp_selector(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lowered = module_name.lower()
    return any(key in lowered for key in ["mlp", "ff", "dense", "proj", "fc"])


@_register_selector("attn")
def _attn_selector(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lowered = module_name.lower()
    return any(key in lowered for key in ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"])


def _compose_selectors(names: Iterable[str]) -> Callable[[str, nn.Module], bool]:
    selectors = [_MODULE_PRESETS[name] for name in names if name in _MODULE_PRESETS]
    if not selectors:
        raise ValueError("No valid module presets specified")

    def combined(name: str, module: nn.Module) -> bool:
        return any(selector(name, module) for selector in selectors)

    return combined


def adapt_hf_model(
    model_name_or_path: str | PreTrainedModel,
    *,
    task: TaskType = "causal-lm",
    max_rank: int = 64,
    modules: Optional[List[str]] = None,
    init: str = "svd",
    trust_remote_code: bool = False,
    device_map: Optional[Dict[str, int]] = None,
):
    """Load (if necessary) and adapt a Hugging Face model with NSN layers."""

    if isinstance(model_name_or_path, str):
        if AutoTokenizer is None:
            raise ImportError("transformers is required to load models")
        model, tokenizer = _load_model_for_task(
            model_name_or_path,
            task=task,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    else:
        model = model_name_or_path
        tokenizer = None

    modules = modules or ["mlp"]
    selector = _compose_selectors(modules)
    model, records = replace_linear_with_nsn(model, selector, max_rank=max_rank)
    if tokenizer is None and hasattr(model, "_tokenizer"):
        tokenizer = getattr(model, "_tokenizer")
    return model, tokenizer, records


def _load_model_for_task(
    model_name_or_path: str,
    *,
    task: TaskType,
    trust_remote_code: bool,
    device_map: Optional[Dict[str, int]],
):
    if AutoTokenizer is None:
        raise ImportError("transformers is required")
    task = task.lower()
    if task == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    elif task == "seq-cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    elif task == "tok-cls":
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    elif task == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    return model, tokenizer


def set_target_flops(model: nn.Module, target_fraction: float) -> int:
    """Set the active rank such that FLOPs roughly match ``target_fraction``."""

    rank = select_rank_for_target_flops(model, target_fraction)
    set_rank(model, rank)
    return rank


__all__ = [
    "adapt_hf_model",
    "set_target_flops",
    "estimate_model_flops",
    "select_rank_for_target_flops",
]
