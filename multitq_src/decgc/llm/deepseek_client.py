"""DeepSeek JSON client used only by the explicit parse-questions command."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from jinja2 import Template
from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class DeepSeekParseError(RuntimeError):
    """Raised after all structured-output attempts have failed."""


def extract_json_object(content: str | None) -> dict[str, Any]:
    """Extract the first valid JSON object from plain or Markdown-wrapped content."""
    if not content or not content.strip():
        raise DeepSeekParseError("DeepSeek returned empty content")
    text = content.strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise DeepSeekParseError("DeepSeek response did not contain a valid JSON object")


class DeepSeekClient:
    def __init__(self, settings: dict, prompts_root: Path):
        api_key = os.getenv(settings["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{settings['api_key_env']} is missing")
        self.settings = settings
        self.consecutive_failures = 0
        self.client = OpenAI(
            api_key=api_key,
            base_url=settings["base_url"],
            timeout=settings["timeout_seconds"],
            max_retries=0,
        )
        self.system = (prompts_root / "question_parser/system.txt").read_text(encoding="utf-8")
        self.template = Template((prompts_root / "question_parser/user.jinja2").read_text(encoding="utf-8"))

    def parse(self, question: str, entities: list[str], relations: list[str], operators: list[str]) -> dict:
        threshold = int(self.settings.get("circuit_breaker_failures", 3))
        if self.consecutive_failures >= threshold:
            raise DeepSeekParseError(
                f"DeepSeek circuit breaker is open after {self.consecutive_failures} consecutive failed questions"
            )
        user = self.template.render(question=question, candidate_entities=entities,
                                    candidate_relations=relations, temporal_operators=operators)
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        primary = self.settings["model"]
        fallback = self.settings.get("fallback_model") or primary
        attempts = max(1, int(self.settings.get("max_retries", 4)))
        errors = []
        for attempt in range(attempts):
            model = primary if attempt < max(1, attempts // 2) else fallback
            json_mode = attempt < attempts - 1
            try:
                arguments = {
                    "model": model,
                    "temperature": self.settings["temperature"],
                    "max_tokens": self.settings["max_tokens"],
                    "messages": messages,
                    "extra_body": {
                        "thinking": {
                            "type": "enabled" if self.settings.get("thinking_enabled", False) else "disabled"
                        }
                    },
                }
                if json_mode:
                    arguments["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**arguments)
                if not response.choices:
                    raise DeepSeekParseError("DeepSeek returned no choices")
                choice = response.choices[0]
                content = choice.message.content
                reasoning_content = getattr(choice.message, "reasoning_content", None)
                usage = getattr(response, "usage", None)
                diagnostics = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_chars": len(content or ""),
                    "reasoning_chars": len(reasoning_content or ""),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
                if not content or not content.strip():
                    raise DeepSeekParseError(f"DeepSeek returned empty content; diagnostics={diagnostics}")
                payload = extract_json_object(content)
                self.consecutive_failures = 0
                LOGGER.info(
                    "DeepSeek structured parse succeeded with %s on attempt %d; diagnostics=%s",
                    model, attempt + 1, diagnostics,
                )
                return payload
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                errors.append(f"attempt={attempt + 1}, model={model}, json_mode={json_mode}, {detail}")
                LOGGER.warning("DeepSeek parse attempt %d/%d failed: %s", attempt + 1, attempts, detail)
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 8))
        self.consecutive_failures += 1
        raise DeepSeekParseError("All DeepSeek parse attempts failed; " + " | ".join(errors))
