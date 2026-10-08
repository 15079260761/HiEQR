"""Chain trajectory summaries."""


def summarize_trajectories(chains: list) -> dict:
    completed = [chain for chain in chains if chain.state == "COMPLETED"]
    return {
        "chains": len(chains), "completed": len(completed),
        "average_stop_hop": sum(chain.hop for chain in completed) / max(1, len(completed)),
        "average_final_reliability": sum(chain.absolute_reliability for chain in completed) / max(1, len(completed)),
    }

