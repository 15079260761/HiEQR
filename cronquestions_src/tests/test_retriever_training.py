import pytest

torch = pytest.importorskip("torch")

from decgc.utils.hashing import state_dict_hash


def test_retriever_like_parameter_actually_updates():
    model = torch.nn.Linear(4, 4)
    before = state_dict_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss = model(torch.ones(2, 4)).pow(2).mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()
    assert state_dict_hash(model) != before
