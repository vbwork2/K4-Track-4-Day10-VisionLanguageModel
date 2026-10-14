"""Check real Transformers conversion rules without downloading model weights."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
bnb = pytest.importorskip("bitsandbytes")

from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.integrations.bitsandbytes import replace_with_bnb_linear
from student.model import _assert_quantization_exclusions, build_quantization_config


def test_real_qwen_quantization_keeps_visual_and_head_unconverted():
    config = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32,
                       "deepstack_visual_indexes": [0, 1]},
    )
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    quantization = build_quantization_config(torch)
    model = replace_with_bnb_linear(
        model, modules_to_not_convert=quantization.llm_int8_skip_modules,
        quantization_config=quantization,
    )
    _assert_quantization_exclusions(model)
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, bnb.nn.Linear4bit)
    assert type(model.model.visual.blocks[0].attn.qkv) is torch.nn.Linear
    assert type(model.model.visual.merger.linear_fc1) is torch.nn.Linear
    assert type(model.lm_head) is torch.nn.Linear
