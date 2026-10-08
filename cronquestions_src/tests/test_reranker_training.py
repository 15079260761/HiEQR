import pytest

torch = pytest.importorskip("torch")

from decgc.utils.hashing import state_dict_hash


def test_reranker_like_bce_updates_parameters():
    model = torch.nn.Linear(3, 1)
    before = state_dict_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    logits = model(torch.randn(4, 3)).squeeze(-1)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0., 1., 0.])).backward()
    assert torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) > 0
    optimizer.step()
    assert state_dict_hash(model) != before
