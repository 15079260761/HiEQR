from decgc.graph.graph_builder import DynamicGraphBuilder
from decgc.schemas import ChainPrefix


def _chain(chain_id, answer, relations, facts=None, state="COMPLETED"):
    return ChainPrefix(chain_id=chain_id, qid="q", hop=1, facts=facts or [], entity_path=["Iraq"],
                       relation_path=relations, time_path=["2007-01-01"], current_entity="Iraq",
                       candidate_answer=answer, state=state)


def test_cooperation_requires_more_than_shared_entity():
    graph = DynamicGraphBuilder().build("q", 1, [_chain("a", "X", ["r"]), _chain("b", "Y", ["r"])], exclusive=False)
    assert not any(edge.edge_type == "COOPERATION" for edge in graph.edges)


def test_competition_is_not_default_full_connection():
    graph = DynamicGraphBuilder().build("q", 1, [_chain("a", "X", ["r"]), _chain("b", "Y", ["s"])], exclusive=False)
    assert not any(edge.edge_type == "COMPETITION" for edge in graph.edges)


def test_pruned_chain_is_excluded():
    graph = DynamicGraphBuilder().build("q", 1, [_chain("p", "X", ["r"], state="PRUNED")])
    assert not graph.nodes

