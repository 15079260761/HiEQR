"""Shared, path-free project constants."""

ACTIVE = "ACTIVE"
COMPLETED = "COMPLETED"
PRUNED = "PRUNED"
CHAIN_STATES = {ACTIVE, COMPLETED, PRUNED}
NODE_TYPES = {"ENTITY", "CHAIN", "ANSWER"}
EDGE_TYPES = {"FACT_EDGE", "CHAIN_MEMBERSHIP", "SUPPORTS_ANSWER", "COOPERATION", "COMPETITION"}
TEMPORAL_OPERATORS = {
    "equal",
    "equal_multi",
    "before",
    "after",
    "before_after",
    "before_last",
    "after_first",
    "first",
    "last",
    "first_last",
}
