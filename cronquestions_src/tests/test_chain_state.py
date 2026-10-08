import pytest

from decgc.graph.chain_state import transition
from decgc.schemas import ChainPrefix


def test_completed_chain_cannot_reactivate():
    chain = ChainPrefix(chain_id="c", qid="q", hop=1, facts=[], entity_path=[], relation_path=[],
                        time_path=[], current_entity="e", state="COMPLETED")
    with pytest.raises(ValueError):
        transition(chain, "ACTIVE")

