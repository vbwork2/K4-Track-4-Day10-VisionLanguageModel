import sys
import types

import pytest

from student import data


def test_public_download_uses_instructor_script_and_validates(monkeypatch, tmp_path):
    source = tmp_path / "public-dataset"
    downloaded = []
    commands = []
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(
        dataset_download=lambda slug: downloaded.append(slug) or str(source),
    ))
    monkeypatch.setattr(data.subprocess, "run", lambda args, **kwargs: commands.append((args, kwargs)))
    monkeypatch.setattr(data, "validate_splits", lambda root, **kwargs: {"validated": str(root)})
    result = data.prepare_public_dataset(tmp_path, tmp_path / "out", tmp_path / "work")
    assert downloaded == ["lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"]
    args, kwargs = commands[0]
    assert args[1] == str(tmp_path / "scripts" / "prepare_data.py")
    assert args[2:4] == ["--source", str(source)]
    assert kwargs["check"] is True
    assert result == {"validated": str(tmp_path / "out")}


def test_existing_split_is_validated_without_redownload(monkeypatch, tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data, "validate_splits", lambda root, **kwargs: {"validated": True})
    monkeypatch.setattr(data.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected preparation"))
    assert data.prepare_public_dataset(tmp_path, tmp_path, tmp_path / "work") == {"validated": True}


def test_rejects_ambiguous_dataset_source(tmp_path):
    with pytest.raises(ValueError, match="either"):
        data.prepare_public_dataset(tmp_path, tmp_path, tmp_path, source_dir="data", zip_path="data.zip")
