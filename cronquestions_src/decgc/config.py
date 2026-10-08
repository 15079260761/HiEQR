"""Recursive YAML composition, interpolation, overrides, validation, and hashing."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils.hashing import object_hash
from .utils.io import write_json

_INTERPOLATION = re.compile(r"\$\{([^}]+)}")


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        value = value[part]
    return value


def _resolve_value(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, root) for item in value]
    if not isinstance(value, str):
        return value
    matches = list(_INTERPOLATION.finditer(value))
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return _lookup(root, matches[0].group(1))
    return _INTERPOLATION.sub(lambda match: str(_lookup(root, match.group(1))), value)


def _interpolate(data: dict[str, Any]) -> dict[str, Any]:
    current = copy.deepcopy(data)
    for _ in range(20):
        resolved = _resolve_value(current, current)
        if resolved == current:
            return resolved
        current = resolved
    raise ValueError("Configuration interpolation did not converge")


def _load_composed(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    seen = seen or set()
    if resolved_path in seen:
        raise ValueError(f"Cyclic config default: {resolved_path}")
    seen.add(resolved_path)
    with resolved_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    defaults = raw.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default in defaults:
        merged = _merge(merged, _load_composed((resolved_path.parent / default).resolve(), seen.copy()))
    return _merge(merged, raw)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    target = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    root: Path
    seed: int = 42


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    env_file: Path
    data_root: Path
    dataset_root: Path
    kg_root: Path
    questions_root: Path
    source_root: Path
    prompts_root: Path
    models_root: Path
    pretrained_retriever: Path
    pretrained_reranker: Path
    outputs_root: Path
    manifests_root: Path
    cache_root: Path
    runs_root: Path
    predictions_root: Path
    evidence_root: Path
    demo_root: Path
    reports_root: Path
    logs_root: Path


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")
    project: ProjectConfig
    paths: PathConfig
    llm: dict[str, Any]
    dataset: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def paths_belong_to_root(self) -> "Config":
        root = self.project.root
        for name, value in self.paths.model_dump().items():
            if root not in Path(value).parents and Path(value) != root:
                raise ValueError(f"paths.{name} is outside project.root")
        return self

    @property
    def hash(self) -> str:
        return object_hash(self.model_dump(mode="json"))

    def prepare_outputs(self) -> None:
        for name, path in self.paths.model_dump().items():
            if name.endswith("_root") and name not in {"data_root", "dataset_root", "kg_root", "questions_root", "source_root", "models_root", "prompts_root"}:
                Path(path).mkdir(parents=True, exist_ok=True)
        for relative in ("question_samples", "question_parses", "fact_embeddings", "kg_indices", "retrieval_pairs",
                         "reranker_pairs", "candidate_chains", "counterfactuals"):
            (self.paths.cache_root / relative).mkdir(parents=True, exist_ok=True)


def load_config(path: Path, overrides: list[str] | None = None) -> tuple[Config, dict[str, Any]]:
    raw = _load_composed(path)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override}")
        key, encoded = override.split("=", 1)
        _set_dotted(raw, key, yaml.safe_load(encoded))
    resolved = _interpolate(raw)
    return Config.model_validate(resolved), resolved


def save_resolved_config(config: Config, raw: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
    write_json(directory / "config_hash.json", {"sha256": config.hash})
