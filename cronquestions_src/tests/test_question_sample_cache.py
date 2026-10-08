from decgc.data.multitq_adapter import MultiTQAdapter
from decgc.data.question_sample_cache import load_or_build_sample


def test_finite_question_sample_is_persisted_and_reused(config):
    adapter = MultiTQAdapter(config)
    first, first_reused = load_or_build_sample(adapter, config.paths.cache_root, "train", 1, 42)
    assert not first_reused
    adapter.load_questions = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full scan"))
    third, third_reused = load_or_build_sample(adapter, config.paths.cache_root, "train", 1, 42)
    assert third_reused
    assert [row.qid for row in third] == [row.qid for row in first]
