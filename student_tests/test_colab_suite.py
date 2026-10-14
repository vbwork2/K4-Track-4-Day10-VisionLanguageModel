"""Verify the Colab suite order and packaging without a GPU or network."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import weakref
import zipfile

import pytest

from student.experiments import run_id_for


ROOT = Path(__file__).resolve().parents[1]


def notebook_cells():
    return json.loads((ROOT / "colab_day10_qlora_resume.ipynb").read_text(encoding="utf-8"))["cells"]


def suite_context(tmp_path, monkeypatch, fail_run=None):
    calls, bundles, cleanups, downloads = [], [], [], []
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: cleanups.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    class Bundle:
        pass

    def fake_run(experiment, repo_root, data_dir, run_root, e4_base="E1", e4_rank=4, **kwargs):
        assert all(reference() is None for reference in bundles)
        run_id = run_id_for(experiment, e4_base if experiment == "E4" else None,
                            e4_rank if experiment == "E4" else None)
        calls.append(run_id)
        if run_id == fail_run:
            raise RuntimeError("simulated interruption")
        run_dir = Path(run_root) / run_id
        run_dir.mkdir(parents=True)
        predictions = run_dir / "predictions_test.jsonl"
        predictions.write_text('{}\n', encoding="utf-8")
        checkpoint = run_dir / "checkpoints" / "step_000300"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text('{}', encoding="utf-8")
        evidence_dir = Path(repo_root) / "report" / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / f"{run_id}.json"
        evidence = {
            "split": "test", "dataset_version": "fixture", "missing_predictions": 0,
            "n_images": 311, "micro": {"f1": 0.3}, "f1_ci95": [0.2, 0.4],
            "meta": {"trainable_params": 100, "peak_vram_gb": 9, "train_minutes": 60},
        }
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        figures = evidence_dir / "figures" / run_id
        figures.mkdir(parents=True)
        figure = figures / "example.jpg"
        figure.write_bytes(run_id.encode())
        evidence.update(evidence_path=str(evidence_path), annotated_examples=[str(figure)])
        bundle = Bundle()
        bundles.append(weakref.ref(bundle))
        return dict(run_id=run_id, predictions=predictions, evidence=evidence, bundle=bundle)

    monkeypatch.setattr("student.experiments.run_experiment", fake_run)
    context = {"Path": Path, "ROOT": tmp_path, "REPO": tmp_path / "repo", "DATA_DIR": tmp_path / "data",
               "RUN_ROOT": tmp_path / "runs", "torch": SimpleNamespace(cuda=cuda),
               "HF_TOKEN": "test-token", "HF_REPO_ID": "owner/private"}
    exec(''.join(notebook_cells()[2]["source"]), context)
    context["PLANNED_RUN_IDS"] = ["e0", "e1", "e2", "e3", "e4_rank_4_base_e1", "e4_rank_64_base_e1"]
    return context, calls, bundles, cleanups, downloads


def test_colab_suite_releases_models_and_packages_all_six_runs(tmp_path, monkeypatch):
    context, calls, bundles, cleanups, downloads = suite_context(tmp_path, monkeypatch)
    cells = notebook_cells()
    exec(''.join(cells[17]["source"]), context)
    exec(''.join(cells[19]["source"]), context)
    exec(''.join(cells[21]["source"]), context)
    assert calls == context["PLANNED_RUN_IDS"]
    assert all(reference() is None for reference in bundles)
    assert len(cleanups) == 7
    assert len(context["ARTIFACTS"]) == 5
    assert downloads == [str(tmp_path / "e0_e4_logs_evidence.zip")]
    with zipfile.ZipFile(downloads[0]) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert not any("checkpoints" in name for name in names)
        assert len([name for name in names if name.startswith("archives/")]) == 5
        for run_id in calls:
            assert archive.read(f"evidence/figures/{run_id}/example.jpg") == run_id.encode()
            assert f"runs/{run_id}/predictions_test.jsonl" in names


def test_colab_suite_keeps_completed_archives_when_next_run_fails(tmp_path, monkeypatch):
    context, calls, bundles, cleanups, downloads = suite_context(tmp_path, monkeypatch, fail_run="e2")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        exec(''.join(notebook_cells()[17]["source"]), context)
    assert calls == ["e0", "e1", "e2"]
    assert [path.name for path in context["ARTIFACTS"]] == ["e0_logs_evidence.zip", "e1_logs_evidence.zip"]
    assert all(path.is_file() for path in context["ARTIFACTS"])
    assert all(reference() is None for reference in bundles)
    assert len(cleanups) == 3
    with pytest.raises(RuntimeError, match="incomplete"):
        exec(''.join(notebook_cells()[21]["source"]), context)
    assert downloads == []


def test_colab_code_cells_compile():
    for index, cell in enumerate(notebook_cells()):
        if cell["cell_type"] == "code":
            compile(''.join(cell["source"]), f"cell-{index}", "exec")
