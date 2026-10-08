from pathlib import Path

import pytest

from decgc.llm.cache import ParseCache
from decgc.llm.local_parser import LocalQuestionParser
from decgc.llm.selective_parser import SelectiveParser


class FailingClient:
    def __init__(self):
        self.calls = 0

    def parse(self, question, entities, relations, operators):
        self.calls += 1
        raise RuntimeError("empty API response")


def test_api_failure_falls_back_locally_and_is_cached(tmp_path: Path):
    client = FailingClient()
    parser = SelectiveParser(LocalQuestionParser(), ParseCache(tmp_path), client=client, threshold=0.8)
    first = parser.parse("q1", "Who interacted?", "entity", {"Iraq"}, {"visit"}, allow_online=True)
    second = parser.parse("q1", "Who interacted?", "entity", {"Iraq"}, {"visit"}, allow_online=True)
    assert first.parse_source == "local_fallback"
    assert second.parse_source == "local_fallback"
    assert client.calls == 1


def test_deepseek_json_extraction_handles_fences_and_rejects_empty():
    pytest.importorskip("openai")
    from jinja2 import Template
    from types import SimpleNamespace

    from decgc.llm.deepseek_client import DeepSeekClient, DeepSeekParseError, extract_json_object

    assert extract_json_object('```json\n{"confidence": 0.9}\n```') == {"confidence": 0.9}
    with pytest.raises(DeepSeekParseError, match="empty"):
        extract_json_object("")

    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content='{"confidence": 0.9}', reasoning_content=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=SimpleNamespace(completion_tokens=5),
            )

    client = object.__new__(DeepSeekClient)
    client.settings = {
        "model": "deepseek-v4-flash", "fallback_model": "deepseek-v4-pro",
        "max_retries": 1, "max_tokens": 1200, "temperature": 0.0,
        "thinking_enabled": False, "circuit_breaker_failures": 3,
    }
    client.consecutive_failures = 0
    client.system = "Return JSON."
    client.template = Template("Question: {{ question }}")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert client.parse("q", [], [], [])["confidence"] == 0.9
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_official_non_answer_annotations_override_conflicting_llm_parse(tmp_path):
    class ConflictingClient:
        def parse(self, question, entities, relations, operators):
            return {
                "topic_entities": ["Japan"],
                "target_entities": ["China"],
                "relation_hints": ["visit"],
                "temporal_constraints": ["2007"],
                "temporal_operator": "equal",
                "answer_type": "time",
                "answer_role": "time",
                "confidence": 1.0,
            }

    from decgc.data.entity_index import EntityIndex

    parser = SelectiveParser(
        LocalQuestionParser(EntityIndex(["Japan", "China"])),
        ParseCache(tmp_path),
        client=ConflictingClient(),
        threshold=0.8,
    )
    parsed = parser.parse(
        "q1",
        "Before the Japanese government, who interacted with PRC?",
        "entity",
        {"Japan", "China"},
        {"visit"},
        allow_online=True,
        metadata={
            "qtype": "before",
            "entity_positions": [
                {"entity": ["Japan"], "position": [7, 30]},
                {"entity": ["China"], "position": [49, 52]},
            ],
        },
    )

    assert parsed.topic_entities == ["China"]
    assert parsed.target_entities == ["Japan"]
    assert parsed.temporal_operator == "before"
    assert parsed.answer_type == "entity"
