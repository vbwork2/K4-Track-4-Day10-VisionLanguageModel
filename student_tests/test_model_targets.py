import pytest

from student.model import discover_lora_targets


def _module_names():
    names = []
    llm_targets = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    vit_targets = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
    for layer in range(36):
        for target in llm_targets:
            names.append((f"model.language_model.layers.{layer}.{target}", object()))
    for layer in range(24):
        for target in vit_targets:
            names.append((f"model.visual.blocks.{layer}.{target}", object()))
    names.extend([
        ("model.language_model.embed_tokens", object()),
        ("model.visual.merger", object()),
        ("model.lm_head", object()),
    ])
    return names


def test_e1_e2_select_exactly_252_llm_targets():
    names = _module_names()
    for setup in ("E1", "E2"):
        llm, vit, selected = discover_lora_targets(names, setup)
        assert len(llm) == 252
        assert len(vit) == 96
        assert selected == llm


def test_e3_selects_252_llm_plus_96_vit_targets():
    llm, vit, selected = discover_lora_targets(_module_names(), "E3")
    assert (len(llm), len(vit), len(selected)) == (252, 96, 348)


def test_architecture_mismatch_fails_before_model_mutation():
    with pytest.raises(RuntimeError, match="Inspect named_modules"):
        discover_lora_targets([("model.language_model.layers.0.self_attn.q_proj", object())], "E1")

