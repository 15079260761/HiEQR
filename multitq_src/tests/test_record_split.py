from decgc.training.record_split import split_records_by_query


def test_record_split_keeps_all_rows_for_a_query_together():
    records = [
        {"query": f"q{query}", "value": row}
        for query in range(20)
        for row in range(3)
    ]

    training, validation = split_records_by_query(records, 0.2)

    training_queries = {row["query"] for row in training}
    validation_queries = {row["query"] for row in validation}
    assert training_queries
    assert validation_queries
    assert training_queries.isdisjoint(validation_queries)
    assert len(training) + len(validation) == len(records)
