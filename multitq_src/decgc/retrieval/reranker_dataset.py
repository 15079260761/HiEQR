"""Binary reranker pair conversion."""


def build_reranker_records(retriever_records: list[dict]) -> list[dict]:
    rows = []
    for record in retriever_records:
        rows.append({"query": record["query"], "text": record["positive"], "label": 1.0})
        rows.append({"query": record["query"], "text": record["negative"], "label": 0.0})
    return rows
