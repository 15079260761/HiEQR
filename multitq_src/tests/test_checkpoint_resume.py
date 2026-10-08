import pytest

torch = pytest.importorskip("torch")

from decgc.training.checkpoint_manager import CheckpointManager


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path)
    path = manager.save("step-1", model, optimizer, None, None,
                        {"epoch": 1, "global_step": 2, "config_hash": "abc"}, best=True)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(10)
    state = manager.load(path, model, optimizer, expected_config_hash="abc")
    assert state["global_step"] == 2
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())
