import json
import random

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

from student.checkpointing import restore_checkpoint, save_checkpoint


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def save_pretrained(self, output_dir, safe_serialization=True):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "adapter_model.safetensors").write_bytes(b"test adapter")


class TinyScaler:
    def __init__(self):
        self.value = 128

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def test_local_full_state_checkpoint_restores_step_optimizer_rng_and_projector(tmp_path):
    random.seed(314)
    numpy.random.seed(314)
    torch.manual_seed(314)
    model = TinyAdapterModel()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = TinyScaler()
    projector_modules = {
        "merger": torch.nn.Linear(2, 2, bias=True),
        "deepstack": torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)]),
    }
    expected_projector = {
        key: {name: value.detach().clone() for name, value in module.state_dict().items()}
        for key, module in projector_modules.items()
    }
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"event": "train_step", "step": 1}) + "\n", encoding="utf-8")

    save_checkpoint(
        run_dir=tmp_path / "e2",
        global_step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        setup="E2",
        config_fingerprint="cfg",
        dataset_version="data",
        trainable_params=14,
        elapsed_seconds=12.5,
        metrics_path=metrics,
        run_manifest={"status": "training"},
        projector_modules=projector_modules,
        remote_required=False,
    )
    checkpoint = tmp_path / "e2" / "checkpoints" / "step_000001"
    expected_rng = (random.random(), numpy.random.random(), torch.rand(1).item())
    for module in projector_modules.values():
        for parameter in module.parameters():
            parameter.data.zero_()
    random.random()
    numpy.random.random()
    torch.rand(1)

    restored = restore_checkpoint(
        checkpoint,
        optimizer,
        scheduler,
        scaler,
        projector_modules=projector_modules,
        active_metrics_path=tmp_path / "restored_metrics.jsonl",
        config_fingerprint="cfg",
        dataset_version="data",
        setup="E2",
    )
    actual_rng = (random.random(), numpy.random.random(), torch.rand(1).item())
    assert restored["global_step"] == 1
    assert restored["elapsed_seconds"] == 12.5
    assert (checkpoint / "projector.pt").is_file()
    assert actual_rng == pytest.approx(expected_rng)
    for key, module in projector_modules.items():
        for name, value in module.state_dict().items():
            assert torch.equal(value, expected_projector[key][name])
    assert (tmp_path / "restored_metrics.jsonl").read_text(encoding="utf-8") == metrics.read_text(encoding="utf-8")
