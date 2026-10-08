"""State transitions kept separate from neural chain representations."""

from ..schemas import ChainPrefix, ChainSnapshot


def transition(chain: ChainPrefix, state: str, reason: str | None = None) -> ChainPrefix:
    if chain.state in {"COMPLETED", "PRUNED"} and state != chain.state:
        raise ValueError(f"Terminal chain {chain.chain_id} cannot transition")
    chain.state = state
    chain.stop_reason = reason
    chain.state_history.append(ChainSnapshot(
        hop=chain.hop, absolute_reliability=chain.absolute_reliability,
        relative_weight=chain.relative_weight, stop_probability=chain.stop_probability, state=state,
    ))
    return chain

