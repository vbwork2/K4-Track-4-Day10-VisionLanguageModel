import pytest

from student.dataset import (
    PROMPT,
    assert_prompt_prefix,
    build_assistant_labels,
    count_image_tokens,
    pad_sequences,
)


def test_prompt_lists_classes_and_coordinate_scale():
    assert all(label in PROMPT for label in ("pothole", "crack", "manhole"))
    assert "0-1000" in PROMPT
    assert "ONLY a JSON array" in PROMPT


def test_prompt_prefix_is_checked_and_length_returned():
    assert assert_prompt_prefix([[1, 2, 3]], [[1, 2, 3, 4, 5]]) == 3
    with pytest.raises(ValueError, match="not a prefix"):
        assert_prompt_prefix([1, 3], [1, 2, 3])


def test_assistant_labels_mask_prompt_and_padding():
    assert build_assistant_labels([[10, 11, 12, 13]], 2, [[1, 1, 1, 0]]) == [[-100, -100, 12, -100]]


def test_pad_sequences_handles_right_and_left_padding():
    values = [[1, 2], [3]]
    assert pad_sequences(values, 0) == [[1, 2], [3, 0]]
    assert pad_sequences(values, 0, padding_side="left") == [[1, 2], [0, 3]]


def test_count_image_tokens_handles_nested_ids():
    assert count_image_tokens([[5, 99, 99], [99, 7]], 99) == 3

