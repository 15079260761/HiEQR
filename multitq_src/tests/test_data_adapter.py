from decgc.data.multitq_adapter import MultiTQAdapter


def test_official_split_is_preserved(config):
    example = MultiTQAdapter(config).load_questions("valid")[0]
    assert example.split == "valid"
    assert example.source_relative_path == "questions/dev.json"
    assert example.qid == "1"

