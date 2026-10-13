"""Qwen3-VL chat encoding and assistant-only loss collation."""

from scripts.labcommon import CLASSES


PROMPT = (
    "Detect every visible road damage object in the image. "
    f"Allowed labels: {', '.join(CLASSES)}. "
    "Return ONLY a JSON array of objects with exactly the keys label and bbox_2d. "
    "Each bbox_2d must contain integer [x1,y1,x2,y2] coordinates normalized to "
    "the range 0-1000 relative to image width and height. "
    "Do not output markdown, explanations, or boxes for unrelated objects. "
    "Return [] if no target object is present."
)

SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "labels",
    "mm_token_type_ids",
    "token_type_ids",
    "position_ids",
}
SEQUENCE_PAD_VALUES = {
    "attention_mask": 0,
    "labels": -100,
    "mm_token_type_ids": 0,
    "token_type_ids": 0,
    "position_ids": 0,
}


def user_message(image, prompt=PROMPT):
    return {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }


def _as_nested_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one encoded example at a time")
        return value[0], True
    return list(value), False


def assert_prompt_prefix(prompt_ids, full_ids):
    """Assert that the complete user prompt is an exact prefix of the chat."""
    prompt, _ = _as_nested_list(prompt_ids)
    full, _ = _as_nested_list(full_ids)
    if full[: len(prompt)] != prompt:
        raise ValueError("Prompt input_ids are not a prefix of the full conversation")
    return len(prompt)


def build_assistant_labels(input_ids, prompt_length, attention_mask=None):
    """Mask prompt and padding tokens while keeping answer tokens trainable."""
    if hasattr(input_ids, "clone"):
        labels = input_ids.clone()
        labels[..., :prompt_length] = -100
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        return labels

    sequences, batched = _as_nested_list(input_ids)
    if attention_mask is None:
        masks = None
    else:
        masks, _ = _as_nested_list(attention_mask)
    labels = [
        -100 if index < prompt_length or (masks is not None and masks[index] == 0) else token
        for index, token in enumerate(sequences)
    ]
    return [labels] if batched else labels


def encode_example(processor, image, answer, prompt=PROMPT):
    """Encode prompt and answer; only assistant answer tokens contribute to loss."""
    user = user_message(image, prompt=prompt)
    prompt_encoding = processor.apply_chat_template(
        [user],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_encoding = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_length = assert_prompt_prefix(prompt_encoding["input_ids"], full_encoding["input_ids"])
    full_encoding["labels"] = build_assistant_labels(
        full_encoding["input_ids"], prompt_length, full_encoding.get("attention_mask")
    )
    return {
        key: value.squeeze(0) if key in SEQUENCE_KEYS and getattr(value, "ndim", 0) == 2 else value
        for key, value in full_encoding.items()
        if hasattr(value, "dtype")
    }


def count_image_tokens(input_ids, image_token_id):
    """Count image placeholder token IDs in a single or batched sequence."""
    values = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids

    def flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    return sum(token == image_token_id for token in flatten(values))


def assert_image_token_count(processor, encoded, expected=220, image_count=1):
    """Fail with a clear diagnostic if the processor does not match the lab's token target."""
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if image_token_id is None or image_token_id == unknown_id:
        raise RuntimeError("Tokenizer does not expose the Qwen3-VL <|image_pad|> token")
    actual = count_image_tokens(encoded["input_ids"], image_token_id)
    target = expected * image_count
    if actual != target:
        raise ValueError(
            f"Expected {target} image tokens for {image_count} image(s), got {actual}; "
            "check processor version, resize policy, and image dimensions"
        )
    return actual


def pad_sequences(sequences, pad_value, padding_side="right"):
    """Pad flat token sequences with Python lists for deterministic CPU validation."""
    if padding_side not in ("left", "right"):
        raise ValueError(f"Unsupported padding side: {padding_side}")
    max_length = max((len(sequence) for sequence in sequences), default=0)
    result = []
    for sequence in sequences:
        padding = [pad_value] * (max_length - len(sequence))
        result.append(padding + list(sequence) if padding_side == "left" else list(sequence) + padding)
    return result


def collate_examples(examples, processor, device=None, padding_side="right"):
    """Pad text fields and concatenate image patches and grid descriptors."""
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    all_keys = set.union(*(set(example) for example in examples))
    for key in all_keys:
        if any(key not in example for example in examples):
            raise ValueError(f"Batch examples have inconsistent processor field: {key}")

    output = {}
    pad_values = dict(SEQUENCE_PAD_VALUES)
    pad_values["input_ids"] = processor.tokenizer.pad_token_id
    if pad_values["input_ids"] is None:
        pad_values["input_ids"] = processor.tokenizer.eos_token_id
    if pad_values["input_ids"] is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    for key in all_keys:
        values = [example[key] for example in examples]
        if key in SEQUENCE_KEYS:
            sequences = [value.tolist() for value in values]
            padded = pad_sequences(sequences, pad_values.get(key, 0), padding_side=padding_side)
            output[key] = torch.tensor(padded, dtype=values[0].dtype)
        else:
            output[key] = torch.cat(values, dim=0)
    if device is not None:
        output = {key: value.to(device, non_blocking=True) for key, value in output.items()}
    return output

