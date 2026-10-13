"""Qwen3-VL quantization, LoRA target discovery, and projector setup."""

from dataclasses import dataclass
import re


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LLM_PATTERN = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
VIT_PATTERN = re.compile(
    r"(?:^|\.)visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)
EXPECTED_ARCHITECTURE_COUNTS = (252, 96)
PROJECTOR_SUFFIXES = ("visual.merger", "visual.deepstack_merger_list")


@dataclass
class ModelBundle:
    model: object
    processor: object
    setup: str
    llm_targets: tuple
    vit_targets: tuple
    projector_modules: dict
    trainable_params: int
    trainable_names: tuple


def discover_lora_targets(named_modules, setup, validate_counts=True):
    """Select only GUIDE-listed LLM/ViT modules and optionally assert architecture counts."""
    if setup == "E0":
        return (), (), ()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError(f"Unsupported LoRA setup: {setup}")
    names = [name for name, _ in named_modules]
    llm = tuple(name for name in names if LLM_PATTERN.search(name))
    vit = tuple(name for name in names if VIT_PATTERN.search(name))
    expected_llm, expected_vit = EXPECTED_ARCHITECTURE_COUNTS
    if validate_counts and (len(llm), len(vit)) != (expected_llm, expected_vit):
        raise RuntimeError(
            f"Unexpected {setup} LoRA target counts: LLM={len(llm)}, ViT={len(vit)}; "
            f"expected LLM={expected_llm}, ViT={expected_vit}. Inspect named_modules() before training."
        )
    return llm, vit, llm + (vit if setup == "E3" else ())


def _first_tensor_dtype(torch, tree):
    if torch.is_tensor(tree) and tree.is_floating_point():
        return tree.dtype
    if isinstance(tree, (tuple, list)):
        for value in tree:
            result = _first_tensor_dtype(torch, value)
            if result is not None:
                return result
    if isinstance(tree, dict):
        for value in tree.values():
            result = _first_tensor_dtype(torch, value)
            if result is not None:
                return result
    return None


def _cast_floating_tree(torch, tree, dtype):
    if torch.is_tensor(tree) and tree.is_floating_point():
        return tree.to(dtype=dtype)
    if isinstance(tree, tuple):
        return tuple(_cast_floating_tree(torch, value, dtype) for value in tree)
    if isinstance(tree, list):
        return [_cast_floating_tree(torch, value, dtype) for value in tree]
    if isinstance(tree, dict):
        return {key: _cast_floating_tree(torch, value, dtype) for key, value in tree.items()}
    return tree


def _make_fp32_island(torch, module):
    class Float32Island(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            output_dtype = _first_tensor_dtype(torch, (args, kwargs))
            if output_dtype is None:
                output_dtype = torch.float16
            args_fp32 = _cast_floating_tree(torch, args, torch.float32)
            kwargs_fp32 = _cast_floating_tree(torch, kwargs, torch.float32)
            device_type = next(self.inner.parameters()).device.type
            with torch.autocast(device_type=device_type, enabled=False):
                output = self.inner(*args_fp32, **kwargs_fp32)
            return _cast_floating_tree(torch, output, output_dtype)

    return Float32Island(module)


def _get_submodule(root, name):
    if hasattr(root, "get_submodule"):
        return root.get_submodule(name)
    current = root
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _find_unique_suffix(root, suffix):
    matches = [(name, module) for name, module in root.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one module ending in {suffix!r}, found {[name for name, _ in matches]}")
    return matches[0]


def _replace_submodule(root, name, replacement):
    parent_name, _, child_name = name.rpartition(".")
    parent = _get_submodule(root, parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _prepare_projectors(torch, model):
    merger_name, merger = _find_unique_suffix(model, "visual.merger")
    deepstack_name, deepstack = _find_unique_suffix(model, "visual.deepstack_merger_list")
    wrapped_merger = _make_fp32_island(torch, merger)
    wrapped_merger.to(dtype=torch.float32)
    _replace_submodule(model, merger_name, wrapped_merger)
    for parameter in wrapped_merger.parameters():
        parameter.requires_grad_(True)

    deepstack_names = []
    for index, child in enumerate(list(deepstack.children())):
        child_name = f"{deepstack_name}.{index}"
        wrapped_child = _make_fp32_island(torch, child)
        wrapped_child.to(dtype=torch.float32)
        _replace_submodule(model, child_name, wrapped_child)
        for parameter in wrapped_child.parameters():
            parameter.requires_grad_(True)
        deepstack_names.append(child_name)
    if not deepstack_names:
        raise RuntimeError("visual.deepstack_merger_list has no merger modules")
    return {
        "merger": _get_submodule(model, merger_name),
        "deepstack": _get_submodule(model, deepstack_name),
    }


def projector_state_dict(projector_modules):
    """Copy full projector state to CPU for a separate durable checkpoint file."""
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in projector_modules.items()
    }


def load_projector_state_dict(projector_modules, state):
    if set(state) != set(projector_modules):
        raise ValueError(f"Projector state keys differ: checkpoint={set(state)}, model={set(projector_modules)}")
    for name, module in projector_modules.items():
        module.load_state_dict(state[name], strict=True)


def build_quantization_config(torch):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def _assert_quantization_exclusions(model):
    import bitsandbytes as bnb

    bad = [
        name
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
        and ("visual" in name.split(".") or name.endswith("lm_head"))
    ]
    if bad:
        raise RuntimeError(f"Quantization touched excluded visual/lm_head modules: {bad[:8]}")


def load_model_bundle(
    setup,
    model_id=MODEL_ID,
    rank=16,
    alpha=None,
    dropout=0.05,
    adapter_path=None,
    trainable=True,
    device_map=None,
):
    """Load QLoRA model and configure the requested E1/E2/E3 training variant."""
    if setup not in ("E0", "E1", "E2", "E3"):
        raise ValueError(f"Unsupported model setup: {setup}")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right" if trainable else "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(torch),
        dtype=torch.float16,
        device_map=device_map or {"": 0},
        low_cpu_mem_usage=True,
    )
    _assert_quantization_exclusions(base)
    llm_targets, vit_targets, selected_targets = discover_lora_targets(base.named_modules(), setup)

    for parameter in base.parameters():
        parameter.requires_grad_(False)
    if setup == "E0":
        model = base
    elif adapter_path:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
    else:
        if alpha is None:
            alpha = rank * 2
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(selected_targets),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, config)

    projector_modules = {}
    if setup in ("E2", "E3"):
        projector_modules = _prepare_projectors(torch, model)
    if setup != "E0" and trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if setup == "E0" and trainable_params != 0:
        raise RuntimeError("E0 must have no trainable parameters")
    if setup == "E1" and any("visual." in name for name in trainable_names):
        raise RuntimeError("E1 must keep the full visual tower and projector frozen")
    if setup in ("E2", "E3"):
        projector_trainables = [name for name in trainable_names if "visual.merger" in name or "visual.deepstack_merger_list" in name]
        if not projector_trainables:
            raise RuntimeError(f"{setup} has no trainable projector parameters")
        if any(parameter.dtype != torch.float32 for module in projector_modules.values() for parameter in module.parameters()):
            raise RuntimeError("Trainable projector parameters must remain float32")
    if setup == "E3" and not any("visual.blocks." in name and "lora_" in name for name in trainable_names):
        raise RuntimeError("E3 has no trainable ViT LoRA parameters")
    if setup != "E0" and not any("lora_" in name for name in trainable_names):
        raise RuntimeError(f"{setup} has no trainable LoRA parameters")

    return ModelBundle(
        model=model,
        processor=processor,
        setup=setup,
        llm_targets=llm_targets,
        vit_targets=vit_targets,
        projector_modules=projector_modules,
        trainable_params=trainable_params,
        trainable_names=trainable_names,
    )

