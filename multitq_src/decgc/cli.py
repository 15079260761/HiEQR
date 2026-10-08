"""Unified command line for the resumable DECGC experiment."""

from __future__ import annotations

import json
import gzip
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from rich.console import Console

from .analysis.error_analysis import error_analysis
from .config import Config, load_config, save_resolved_config
from .data.entity_index import EntityIndex
from .data.inspector import inspect_data as run_inspection
from .data.kg_index import KGIndex
from .data.multitq_adapter import MultiTQAdapter
from .data.question_sample_cache import load_or_build_sample
from .data.sampler import stratified_sample
from .data.source_signature import directory_signature, file_signature
from .evaluation.demo_report import generate_report
from .evaluation.answer_normalizer import normalize_answer
from .evaluation.evidence_metrics import evidence_metrics
from .evaluation.evidence_validator import validate_chain
from .evaluation.evaluator import aggregate_completed, stop_candidates
from .evaluation.metrics import (
    answer_diagnostics,
    answer_diagnostics_iter,
    answer_metrics,
)
from .evaluation.prediction_writer import chain_evidence, write_prediction_bundle
from .evaluation.quality_gate import evaluate_metric_gate
from .graph.graph_builder import DynamicGraphBuilder
from .llm.cache import ParseCache
from .llm.deepseek_client import DeepSeekClient
from .llm.local_parser import LocalQuestionParser
from .llm.selective_parser import SelectiveParser
from .llm.validator import validate_parse
from .models.decgc_model import DECGCModel
from .retrieval.beam_search import TemporalBeamSearch
from .retrieval.bge_encoder import BGEEncoder
from .retrieval.candidate_cache import (
    chain_shard_files,
    chain_shard_is_current,
    chain_shard_path,
    write_chain_cache,
)
from .retrieval.counterfactual_builder import build_counterfactuals as make_counterfactuals
from .retrieval.component_ablation import (
    NO_CROSS_ENCODER_RERANKER,
    NO_DENSE_RETRIEVER,
    AnswerBlindLexicalEncoder,
    candidate_chain_root,
    candidate_generation_variant,
)
from .retrieval.fact_formatter import format_fact
from .retrieval.fact_index import FactIndex
from .retrieval.symbolic_temporal import (
    direct_temporal_candidates,
    merge_symbolic_answers,
)
from .retrieval.type_consistency import answer_type_score
from .retrieval.reranker_dataset import build_reranker_records
from .retrieval.reranker_model import BGEReranker
from .retrieval.reranker_trainer import train_reranker as fit_reranker
from .retrieval.retriever_dataset import (
    RETRIEVER_DATA_VERSION,
    build_retriever_records_for_example,
)
from .retrieval.retriever_trainer import train_retriever as fit_retriever
from .schemas import ChainPrefix, ChainSnapshot, StructuredQuery
from .training.checkpoint_cleanup import cleanup_run
from .training.checkpoint_manager import CheckpointManager
from .training.optimizer import build_optimizer
from .training.reasoner_trainer import train_reasoner as fit_reasoner
from .training.reasoner_dataset import (
    collate_question_batches,
    is_optimal_stop_prefix,
    select_balanced_prefixes,
)
from .utils.device import select_device
from .utils.artifact_cleanup import (
    obsolete_artifact_candidates,
    remove_scoped_paths,
)
from .utils.hashing import file_hash, object_hash, state_dict_hash
from .utils.io import (
    append_jsonl,
    atomic_text,
    read_json,
    read_json_gzip,
    write_json,
    write_json_gzip,
    write_jsonl,
    write_text_lines,
)
from .utils.logging import configure_logging
from .utils.seed import seed_everything
from .utils.stage_cache import completed_transformer_training, stage_config_hash

app = typer.Typer(no_args_is_help=True, rich_markup_mode="markdown")
console = Console()
ConfigPath = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
EVALUATION_VERSION = 15


def _context(config_path: Path, seed: int | None, log_level: str, set_values: list[str] | None = None) -> tuple[Config, dict]:
    configure_logging(log_level)
    overrides = list(set_values or [])
    if seed is not None:
        overrides.append(f"project.seed={seed}")
    config, raw = load_config(config_path, overrides)
    seed_everything(config.project.seed)
    config.prepare_outputs()
    return config, raw


def _run_dir(config: Config, stage: str) -> Path:
    experiment = str(
        config.model_extra.get("experiment", {}).get("name", "unnamed-experiment")
    ).strip()
    if not experiment or any(character in experiment for character in "/\\"):
        raise ValueError("experiment.name must be a non-empty directory name")
    return (
        config.paths.runs_root
        / stage
        / experiment
        / f"multitq-{stage_config_hash(config, stage)[:12]}"
    )


def _legacy_run_dir(config: Config, stage: str) -> Path:
    """Locate loss-agnostic checkpoints produced before named run folders."""
    return (
        config.paths.runs_root
        / stage
        / f"multitq-{stage_config_hash(config, stage)[:12]}"
    )


def _reasoner_checkpoint_root(config: Config) -> Path:
    """Resolve an explicitly frozen ablation reasoner or the matched default."""
    study = config.model_extra.get("ablation_study", {})
    explicit_run = study.get("reasoner_run_dir")
    if explicit_run is None:
        return _run_dir(config, "reasoner") / "checkpoints"
    run_dir = Path(str(explicit_run))
    nested = run_dir / "checkpoints"
    if nested.is_dir() or not (run_dir / "best.json").exists():
        return nested
    return run_dir


def _kg_index(config: Config) -> KGIndex:
    path = config.paths.cache_root / "kg_indices" / "multitq_full.pkl"
    if not path.exists():
        raise RuntimeError("KG index missing; run decgc build-kg-index first")
    return KGIndex.load(path)


def _kg_catalog(config: Config) -> tuple[set[str], set[str]]:
    """Load the small entity/relation catalog without unpickling 461k facts."""
    directory = config.paths.cache_root / "kg_indices"
    entities_path = directory / "multitq_full.entities.json"
    relations_path = directory / "multitq_full.relations.json"
    if entities_path.exists() and relations_path.exists():
        return set(read_json(entities_path)), set(read_json(relations_path))
    index = _kg_index(config)
    entities = set(index.adjacency)
    relations = {fact.relation_id for fact in index.facts.values()}
    write_json(entities_path, sorted(entities))
    write_json(relations_path, sorted(relations))
    return entities, relations


def _parse_path(config: Config, split: str) -> Path:
    return config.paths.cache_root / "question_parses" / f"{split}.jsonl"


def _parse_repair_path(config: Config, split: str) -> Path:
    return config.paths.cache_root / "question_parses" / f"{split}.local_repair.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _parsed(config: Config, split: str) -> dict[str, StructuredQuery]:
    path = _parse_path(config, split)
    repair_path = _parse_repair_path(config, split)
    if not path.exists() and not repair_path.exists():
        raise RuntimeError(f"Question parse cache missing: {path}")
    queries: dict[str, StructuredQuery] = {}
    # Local repairs are loaded first. A later complete primary/LLM parse always
    # takes precedence, so deterministic fallback never overwrites richer data.
    for source in (repair_path, path):
        if not source.exists():
            continue
        for row in _read_jsonl(source):
            query = StructuredQuery.model_validate(row)
            queries[str(query.qid)] = query
    return queries


def _ensure_local_parse_cache(config: Config, split: str, examples) -> None:
    """Atomically fill only missing parses with the deterministic local parser.

    Existing LLM parses are preserved verbatim. Normalizing every qid to text
    prevents JSON numeric/string representation differences from causing a
    late KeyError after expensive candidate-chain construction.
    """
    path = _parse_path(config, split)
    repair_path = _parse_repair_path(config, split)
    try:
        existing = _parsed(config, split)
    except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError):
        existing = {}
    missing = [
        example for example in examples
        if str(example.qid) not in existing
    ]
    if not missing:
        return

    repairs: dict[str, dict] = {}
    if repair_path.exists():
        for row in _read_jsonl(repair_path):
            query = StructuredQuery.model_validate(row)
            repairs[str(query.qid)] = query.model_dump(mode="json")
    entities, relations = _kg_catalog(config)
    local = LocalQuestionParser(EntityIndex(list(entities)), relations)
    for example in missing:
        parsed = local.parse(
            str(example.qid),
            example.question,
            example.answer_type,
            {
                key: value
                for key, value in example.raw_record.items()
                if key != "answers"
            },
        )
        payload = parsed.model_dump(mode="json")
        repairs[str(parsed.qid)] = payload
        existing[str(parsed.qid)] = parsed
    # Never touch the primary parse file: candidate-chain manifests fingerprint
    # it, and rewriting it would invalidate every otherwise reusable shard.
    write_jsonl(repair_path, repairs.values())
    selected_qids = [str(example.qid) for example in examples]
    write_json(repair_path.with_suffix(".manifest.json"), {
        "split": split,
        "questions": len(repairs),
        "combined_questions": len(existing),
        "required_questions": len(selected_qids),
        "selection_hash": object_hash(selected_qids),
        "parse_sources": dict(Counter(
            row.get("parse_source", "unknown") for row in repairs.values()
        )),
        "auto_completed_for_evaluation": True,
        "config_hash": config.hash,
    })
    console.print(
        f"Auto-completed {len(missing)} missing {split} parses; "
        f"combined cache now contains {len(existing)} questions; "
        f"primary cache left unchanged at {path}"
    )


def _best_model(config: Config, stage: str, fallback: Path) -> Path:
    candidate = _run_dir(config, stage) / "best_model"
    if candidate.exists():
        return candidate
    legacy = _legacy_run_dir(config, stage) / "best_model"
    return legacy if legacy.exists() else fallback


def _examples(config: Config, split: str, limit: int | None = None):
    demo = config.model_extra.get("demo", {})
    if limit is None and demo.get("enabled"):
        key = {"train": "train_max_samples", "valid": "valid_max_samples", "test": "evaluation_max_samples"}[split]
        limit = demo.get(key)
    seed = int(demo.get("sampling_seed", config.project.seed))
    adapter = MultiTQAdapter(config)
    if limit is not None:
        examples, reused = load_or_build_sample(adapter, config.paths.cache_root, split, int(limit), seed)
        action = "Reusing" if reused else "Created"
        console.print(f"{action} cached {split} sample ({len(examples)} questions)")
        return examples
    return stratified_sample(adapter.load_questions(split), limit, seed)


def _common_options(config: ConfigPath, resume: str, limit: int | None, seed: int | None, log_level: str):
    return _context(config, seed, log_level)


@app.command()
def doctor(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
           log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Validate offline models, data, GPU, disk, API configuration, and write access."""
    cfg, raw = _context(config, seed, log_level, set_values)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    load_dotenv(cfg.paths.env_file, override=True)
    checks: dict[str, object] = {}
    checks["project_root"] = cfg.project.root.exists()
    checks["data"] = cfg.paths.dataset_root.exists()
    checks["env_file"] = cfg.paths.env_file.exists()
    checks["api_key_present"] = bool(os.getenv(cfg.llm["api_key_env"]))
    checks["python"] = sys.version
    checks["torch"] = torch.__version__
    try:
        import transformers
        checks["transformers"] = transformers.__version__
    except ImportError:
        checks["transformers"] = "missing"
    checks["cuda_available"] = torch.cuda.is_available()
    checks["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        checks["gpu"] = torch.cuda.get_device_name(0)
        checks["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    checks["disk_free_gb"] = round(shutil.disk_usage(cfg.project.root).free / 2**30, 2)
    checks["minimum_free_disk_gb"] = float(
        cfg.model_extra.get("checkpoint", {}).get(
            "minimum_free_disk_gb",
            0,
        )
    )
    checks["disk_space_ok"] = (
        checks["disk_free_gb"] >= checks["minimum_free_disk_gb"]
    )
    for name, path in {"retriever": cfg.paths.pretrained_retriever, "reranker": cfg.paths.pretrained_reranker}.items():
        files = {
            "directory": path.is_dir(), "config": (path / "config.json").exists(),
            "tokenizer": any((path / item).exists() for item in ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model")),
            "weights": any((path / item).exists() for item in ("model.safetensors", "pytorch_model.bin")),
        }
        checks[f"{name}_files"] = files
    try:
        retriever = BGEEncoder(cfg.paths.pretrained_retriever).to(select_device()).eval()
        with torch.inference_mode():
            checks["retriever_probe_shape"] = list(retriever.encode(["temporal fact"]).shape)
        reranker = BGEReranker(cfg.paths.pretrained_reranker).to(select_device()).eval()
        with torch.inference_mode():
            checks["reranker_probe_score"] = float(reranker.score(["question"], ["candidate fact"])[0])
        checks["offline_model_load"] = True
    except Exception as error:
        checks["offline_model_load"] = False
        checks["model_error"] = f"{type(error).__name__}: {error}"
    probe = cfg.paths.outputs_root / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    checks["outputs_writable"] = True
    required = [checks["project_root"], checks["data"], checks["env_file"], checks["offline_model_load"],
                checks["outputs_writable"], checks["disk_space_ok"],
                checks["retriever_files"]["directory"],
                checks["retriever_files"]["config"], checks["retriever_files"]["tokenizer"],
                checks["retriever_files"]["weights"], checks["reranker_files"]["directory"],
                checks["reranker_files"]["config"], checks["reranker_files"]["tokenizer"],
                checks["reranker_files"]["weights"]]
    if cfg.llm.get("enabled"):
        required.append(checks["api_key_present"])
    checks["ok"] = all(required)
    write_json(cfg.paths.manifests_root / "doctor_report.json", checks)
    save_resolved_config(cfg, raw, cfg.paths.manifests_root)
    console.print_json(data=checks)
    if not checks["ok"]:
        raise typer.Exit(1)


@app.command("inspect-data")
def inspect_data(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
                 log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Inspect actual files without modifying source data."""
    cfg, raw = _context(config, seed, log_level, set_values)
    result = run_inspection(cfg, resume=resume != "none")
    save_resolved_config(cfg, raw, cfg.paths.manifests_root)
    console.print(f"Inventoried {len(result['files'])} files")


@app.command("build-kg-index")
def build_kg_index(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
                   log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Build provenance-preserving bidirectional adjacency."""
    cfg, _ = _context(config, seed, log_level, set_values)
    target = cfg.paths.cache_root / "kg_indices" / "multitq_full.pkl"
    manifest_path = target.with_suffix(".manifest.json")
    adapter = MultiTQAdapter(cfg)
    source = file_signature(adapter.kg_path("full"))
    if resume != "none" and target.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("source") == source and manifest.get("limit") == limit:
            console.print(f"KG cache hit: reusing {target} ({manifest['facts']} facts; source unchanged)")
            return
        if (
            "source" not in manifest
            and manifest.get("facts")
            and limit is None
        ):
            # Upgrade caches produced by DECGC <=0.2.1 without reparsing the TSV.
            index = KGIndex.load(target)
            write_json(target.parent / "multitq_full.entities.json", sorted(index.adjacency))
            write_json(target.parent / "multitq_full.relations.json",
                       sorted({fact.relation_id for fact in index.facts.values()}))
            write_json(manifest_path, {**manifest, "source": source, "limit": None})
            console.print(
                f"Upgraded and reused existing KG cache ({len(index.facts)} facts); "
                "the source TSV was not rebuilt"
            )
            return
    facts = list(adapter.iter_facts("full", limit))
    index = KGIndex(facts)
    index.save(target)
    directory = target.parent
    write_json(directory / "multitq_full.entities.json", sorted(index.adjacency))
    write_json(directory / "multitq_full.relations.json",
               sorted({fact.relation_id for fact in facts}))
    write_json(manifest_path, {
        "data_hash": index.data_hash, "facts": len(facts), "source": source,
        "limit": limit, "config_hash": cfg.hash,
    })
    console.print(f"Indexed {len(facts)} facts")


@app.command("parse-questions")
def parse_questions(config: ConfigPath, split: str = "all", resume: str = "none", limit: int | None = None,
                    seed: int | None = None, log_level: str = "INFO",
                    no_online: bool = typer.Option(False, "--no-online"),
                    set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Parse questions; this is the only command allowed to call DeepSeek."""
    cfg, _ = _context(config, seed, log_level, set_values)
    entities, relations = _kg_catalog(cfg)
    entity_index = EntityIndex(list(entities))
    local = LocalQuestionParser(entity_index, relations)
    client = None
    if cfg.llm["enabled"] and not no_online:
        load_dotenv(cfg.paths.env_file, override=True)
        try:
            client = DeepSeekClient(cfg.llm, cfg.paths.prompts_root)
        except Exception as error:
            console.print(
                f"[yellow]DeepSeek client unavailable; continuing with local parsing: "
                f"{type(error).__name__}: {error}[/yellow]"
            )
    parser = SelectiveParser(local, ParseCache(cfg.paths.cache_root / "question_parses" / "items"),
                             client, float(cfg.llm["parse_confidence_threshold"]))
    splits = ["train", "valid", "test"] if split == "all" else [split]
    for current in splits:
        output = _parse_path(cfg, current)
        selected_examples = _examples(cfg, current, limit)
        selection_hash = object_hash([example.qid for example in selected_examples])
        manifest_path = output.with_suffix(".manifest.json")
        if resume != "none" and output.exists() and manifest_path.exists():
            existing_manifest = read_json(manifest_path)
            if (
                existing_manifest.get("prompt_version") == parser.version
                and existing_manifest.get("selection_hash") == selection_hash
            ):
                continue
        partial = output.with_suffix(".partial.jsonl")
        partial_manifest = output.with_suffix(".partial.manifest.json")
        partial_is_current = False
        if resume != "none" and partial.exists() and partial_manifest.exists():
            partial_metadata = read_json(partial_manifest)
            partial_is_current = (
                partial_metadata.get("prompt_version") == parser.version
                and partial_metadata.get("selection_hash") == selection_hash
            )
        if not partial_is_current:
            atomic_text(partial, "")
            write_json(partial_manifest, {
                "prompt_version": parser.version,
                "selection_hash": selection_hash,
            })
            completed_qids = set()
        else:
            completed_qids = {str(row["qid"]) for row in _read_jsonl(partial)}
        for example in selected_examples:
            if str(example.qid) in completed_qids:
                continue
            parsed = parser.parse(
                example.qid, example.question, example.answer_type, entities, relations,
                allow_online=not no_online,
                metadata={
                    key: value
                    for key, value in example.raw_record.items()
                    if key != "answers"
                },
            )
            append_jsonl(partial, parsed.model_dump(mode="json"))
            completed_qids.add(str(example.qid))
        partial.replace(output)
        partial_manifest.unlink(missing_ok=True)
        final_rows = _read_jsonl(output)
        source_counts = dict(Counter(row["parse_source"] for row in final_rows))
        parse_quality = {
            "topic_entity_coverage": (
                sum(bool(row.get("topic_entities")) for row in final_rows) / len(final_rows)
                if final_rows else 0.0
            ),
            "relation_coverage": (
                sum(bool(row.get("relation_hints")) for row in final_rows) / len(final_rows)
                if final_rows else 0.0
            ),
            "temporal_operator_coverage": (
                sum(bool(row.get("temporal_operator")) for row in final_rows) / len(final_rows)
                if final_rows else 0.0
            ),
            "operators": dict(Counter(
                row.get("temporal_operator") or "none" for row in final_rows
            )),
            "granularities": dict(Counter(
                row.get("temporal_granularity") or "unknown" for row in final_rows
            )),
            "answer_roles": dict(Counter(
                row.get("answer_role") or "unknown" for row in final_rows
            )),
            "relations": dict(Counter(
                relation
                for row in final_rows
                for relation in row.get("relation_hints", [])
            )),
        }
        write_json(manifest_path, {
            "split": current,
            "questions": len(final_rows),
            "parse_sources": source_counts,
            "parse_quality": parse_quality,
            "prompt_version": parser.version,
            "selection_hash": selection_hash,
            "online_allowed": not no_online,
            "config_hash": cfg.hash,
        })
        console.print(f"Parsed {len(final_rows)} {current} questions: {source_counts}")
        console.print_json(data=parse_quality)


@app.command("probe-llm")
def probe_llm(config: ConfigPath, resume: str = "none", limit: int | None = None,
              seed: int | None = None, log_level: str = "INFO",
              set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Make one safe structured DeepSeek request without exposing the API key."""
    cfg, _ = _context(config, seed, log_level, set_values)
    loaded = load_dotenv(cfg.paths.env_file, override=True)
    if not loaded:
        raise RuntimeError(f"Could not load environment file: {cfg.paths.env_file}")
    client = DeepSeekClient(cfg.llm, cfg.paths.prompts_root)
    payload = validate_parse(
        client.parse(
            "Who visited Iraq on 27 February 2007?",
            ["Iraq"],
            ["visit"],
            ["equal"],
        ),
        {"Iraq"},
        {"visit"},
        {"equal"},
    )
    report = {
        "ok": True,
        "provider": cfg.llm["provider"],
        "base_url": cfg.llm["base_url"],
        "model": cfg.llm["model"],
        "thinking_enabled": cfg.llm.get("thinking_enabled", False),
        "api_key_present": bool(os.getenv(cfg.llm["api_key_env"])),
        "validated_payload": payload,
    }
    write_json(cfg.paths.manifests_root / "deepseek_probe.json", report)
    console.print_json(data=report)


@app.command("verify-parse-quality")
def verify_parse_quality(
    config: ConfigPath,
    seed: int | None = None,
    log_level: str = "INFO",
    set_values: list[str] = typer.Option(None, "--set"),
) -> None:
    """Record structured-query coverage and optionally stop before training."""
    cfg, _ = _context(config, seed, log_level, set_values)
    settings = cfg.model_extra.get("quality_gate", {})
    actual: dict[str, float] = {}
    required: dict[str, float] = {}
    for split in ("train", "valid"):
        manifest_path = _parse_path(cfg, split).with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise RuntimeError(
                f"Parse manifest is missing: {manifest_path}"
            )
        quality = read_json(manifest_path).get("parse_quality", {})
        for metric, setting in (
            ("topic_entity_coverage", "parse_topic_entity_min"),
            ("relation_coverage", "parse_relation_min"),
            ("temporal_operator_coverage", "parse_temporal_operator_min"),
        ):
            name = f"{split}_{metric}"
            actual[name] = float(quality.get(metric, 0.0))
            required[name] = float(settings.get(setting, 0.0))
    result = evaluate_metric_gate(actual, required)
    result["source"] = "persisted_parse_manifests"
    blocking = bool(settings.get("blocking", True))
    result["blocking"] = blocking
    write_json(cfg.paths.manifests_root / "parse_quality_gate.json", result)
    console.print_json(data=result)
    if not result["passed"]:
        if blocking:
            raise typer.Exit(code=2)
        console.print(
            "[yellow]Parse-quality targets were not met; continuing because "
            "quality_gate.blocking=false.[/yellow]"
        )


@app.command("build-retriever-data")
def build_retriever_data(config: ConfigPath, resume: str = "none", limit: int | None = None,
                         seed: int | None = None, log_level: str = "INFO",
                         set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Construct weak positives and neighborhood hard negatives after candidate generation."""
    cfg, _ = _context(config, seed, log_level, set_values)
    output = cfg.paths.cache_root / "retrieval_pairs" / "train.jsonl"
    examples = _examples(cfg, "train", limit)
    settings = cfg.model_extra.get("retriever_data", {})
    max_pairs = int(settings.get("max_pairs_per_question", 4))
    max_neighbors = int(settings.get("max_neighbor_facts_scanned", 4096))
    progress_every = max(1, int(settings.get("progress_every", 250)))
    minimum_coverage = float(
        cfg.model_extra.get("quality_gate", {}).get(
            "retriever_question_coverage_min",
            0.0,
        )
    )
    kg_manifest = read_json(cfg.paths.cache_root / "kg_indices" / "multitq_full.manifest.json")
    expected_manifest = {
        "data_version": RETRIEVER_DATA_VERSION,
        "kg_data_hash": kg_manifest["data_hash"],
        "parse_source": file_signature(_parse_path(cfg, "train")),
        "question_ids_hash": object_hash([example.qid for example in examples]),
        "questions": len(examples),
        "max_pairs_per_question": max_pairs,
        "max_neighbor_facts_scanned": max_neighbors,
    }
    manifest_path = output.with_suffix(".manifest.json")
    if resume != "none" and output.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected_manifest.items()):
            console.print(f"Retriever-data cache hit: reusing {manifest['records']} triples in {output}")
            cached_coverage = float(manifest.get("question_coverage", 0.0))
            if cached_coverage < minimum_coverage:
                message = (
                    "Cached retriever-data question coverage is insufficient: "
                    f"{cached_coverage:.3%} < {minimum_coverage:.3%}"
                )
                if bool(
                    cfg.model_extra.get("quality_gate", {}).get(
                        "blocking",
                        True,
                    )
                ):
                    raise RuntimeError(message)
                console.print(
                    f"[yellow]{message}; continuing because "
                    "quality_gate.blocking=false.[/yellow]"
                )
            remove_scoped_paths(
                list(output.parent.glob("train-shards-*")),
                cfg.paths.outputs_root,
                execute=True,
            )
            return

    shard_signature = object_hash(expected_manifest)[:16]
    shard_root = output.parent / f"train-shards-{shard_signature}"
    shard_root.mkdir(parents=True, exist_ok=True)
    kg_index = _kg_index(cfg)
    parsed_queries = _parsed(cfg, "train")
    started = time.monotonic()
    records: list[dict] = []
    reused_shards = 0
    questions_with_records = 0
    for index, example in enumerate(examples):
        shard = shard_root / f"{index:08d}.jsonl"
        if resume != "none" and shard.exists():
            rows = _read_jsonl(shard)
            reused_shards += 1
        else:
            rows = build_retriever_records_for_example(
                example,
                kg_index,
                parsed_queries[str(example.qid)],
                max_pairs_per_question=max_pairs,
                max_neighbor_facts_scanned=max_neighbors,
            )
            write_jsonl(shard, rows)
        if rows:
            questions_with_records += 1
        records.extend(rows)
        processed = index + 1
        if processed == len(examples) or processed % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = processed / elapsed
            remaining = (len(examples) - processed) / rate if rate else 0.0
            console.print(
                f"Retriever data: {processed}/{len(examples)} questions, {len(records)} triples, "
                f"{reused_shards} resumed ({rate:.1f} questions/s, ETA {remaining:.0f}s)"
            )
    if not records:
        raise RuntimeError(
            "No retriever triples were constructed; verify topic-entity parses and answer normalization"
        )
    write_jsonl(output, records)
    write_json(manifest_path, {
        **expected_manifest,
        "records": len(records),
        "questions_with_records": questions_with_records,
        "question_coverage": questions_with_records / len(examples),
        "resolved_config_hash": cfg.hash,
        "shard_signature": shard_signature,
    })
    cleanup = remove_scoped_paths(
        list(output.parent.glob("train-shards-*")),
        cfg.paths.outputs_root,
        execute=True,
    )
    if cleanup["removed"]:
        console.print(
            f"Removed {len(cleanup['removed'])} completed retriever shard "
            f"directories ({cleanup['bytes_to_release'] / 2**20:.1f} MiB)"
        )
    question_coverage = questions_with_records / len(examples)
    console.print(
        f"Wrote {len(records)} retriever triples covering "
        f"{questions_with_records}/{len(examples)} questions "
        f"({question_coverage:.1%})"
    )
    if question_coverage < minimum_coverage:
        message = (
            "Retriever-data question coverage is insufficient: "
            f"{question_coverage:.3%} < {minimum_coverage:.3%}"
        )
        if bool(
            cfg.model_extra.get("quality_gate", {}).get(
                "blocking",
                True,
            )
        ):
            raise RuntimeError(message)
        console.print(
            f"[yellow]{message}; continuing because "
            "quality_gate.blocking=false.[/yellow]"
        )


@app.command("train-retriever")
def train_retriever(config: ConfigPath, resume: str = "auto", limit: int | None = None, seed: int | None = None,
                    log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Fine-tune all active BGE retriever parameters and verify hashes/gradients."""
    cfg, raw = _context(config, seed, log_level, set_values)
    run_dir = _run_dir(cfg, "retriever")
    completed = (
        completed_transformer_training(run_dir)
        if resume != "none"
        else None
    )
    if completed is not None:
        console.print(f"Retriever-training cache hit: reusing {run_dir}")
        console.print_json(data=completed)
        return
    records = _read_jsonl(cfg.paths.cache_root / "retrieval_pairs" / "train.jsonl")
    if limit:
        records = records[:limit]
    model = BGEEncoder(cfg.paths.pretrained_retriever)
    result = fit_retriever(model, records, run_dir,
                           cfg.model_extra["retriever_training"], select_device(), resume)
    if cfg.model_extra["checkpoint"]["cleanup_after_successful_save"]:
        cleanup_run(run_dir / "checkpoints",
                    int(cfg.model_extra["checkpoint"]["keep_last"]), execute=True)
    save_resolved_config(cfg, raw, run_dir)
    console.print_json(data=result)


@app.command("encode-facts")
def encode_facts(config: ConfigPath, retriever_checkpoint: str = "best", resume: str = "none",
                 limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
                 set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Re-encode Demo KG facts with the fine-tuned retriever."""
    cfg, _ = _context(config, seed, log_level, set_values)
    model_path = _best_model(cfg, "retriever", cfg.paths.pretrained_retriever)
    directory = cfg.paths.cache_root / "fact_embeddings"
    manifest_path = directory / "manifest.json"
    kg_manifest = read_json(cfg.paths.cache_root / "kg_indices" / "multitq_full.manifest.json")
    artifact_signature = directory_signature(model_path)
    if resume != "none" and all((directory / name).exists() for name in ("embeddings.npy", "ids.json", "manifest.json")):
        existing = read_json(manifest_path)
        if (
            existing.get("kg_data_hash") == kg_manifest.get("data_hash")
            and existing.get("retriever_artifact_signature") == artifact_signature
            and existing.get("limit") == limit
        ):
            console.print(f"Fact-embedding cache hit: reusing {existing['facts']} vectors in {directory}")
            return
    model = BGEEncoder(model_path).to(select_device()).eval()
    index = _kg_index(cfg)
    facts = list(index.facts.values())
    if limit is not None:
        facts = facts[:limit]
    import numpy as np

    hidden_size = int(model.model.config.hidden_size)
    partial_path = directory / "embeddings.partial.npy"
    partial_manifest_path = directory / "embeddings.partial.manifest.json"
    partial_expected = {
        "encoding_cache_version": 1,
        "retriever_artifact_signature": artifact_signature,
        "kg_data_hash": kg_manifest["data_hash"],
        "facts": len(facts),
        "limit": limit,
        "hidden_size": hidden_size,
    }
    completed_facts = 0
    partial_metadata = {}
    if resume != "none" and partial_manifest_path.exists():
        partial_metadata = read_json(partial_manifest_path)
        if all(partial_metadata.get(key) == value for key, value in partial_expected.items()):
            completed_facts = int(partial_metadata.get("completed_facts", 0))
        else:
            partial_metadata = {}

    final_embeddings_path = directory / "embeddings.npy"
    if (
        completed_facts == len(facts)
        and not partial_path.exists()
        and not final_embeddings_path.exists()
    ):
        completed_facts = 0
    if (
        completed_facts == len(facts)
        and final_embeddings_path.exists()
        and not partial_path.exists()
    ):
        embeddings = np.load(final_embeddings_path, mmap_mode="r")
        if embeddings.shape != (len(facts), hidden_size):
            completed_facts = 0
        del embeddings

    memory_map = None
    if completed_facts < len(facts):
        if completed_facts and partial_path.exists():
            candidate = np.load(partial_path, mmap_mode="r+")
            if candidate.shape == (len(facts), hidden_size):
                memory_map = candidate
                candidate = None
            else:
                del candidate
                completed_facts = 0
        if memory_map is None:
            memory_map = np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(facts), hidden_size),
            )
            completed_facts = 0
            write_json(partial_manifest_path, {**partial_expected, "completed_facts": 0})

    total_batches = (len(facts) + 127) // 128
    encode_started = time.monotonic()
    with torch.inference_mode():
        for start in range(completed_facts, len(facts), 128):
            end = min(start + 128, len(facts))
            encoded = model.encode([format_fact(item) for item in facts[start:end]]).detach().cpu().numpy()
            memory_map[start:end] = encoded
            memory_map.flush()
            completed_facts = end
            write_json(partial_manifest_path, {
                **partial_expected,
                "completed_facts": completed_facts,
            })
            batch_number = (end + 127) // 128
            if batch_number == total_batches or batch_number % 100 == 0:
                elapsed = max(time.monotonic() - encode_started, 1e-6)
                processed_batches = max(1, (completed_facts + 127) // 128)
                resumed_batches = (partial_metadata.get("completed_facts", 0) + 127) // 128
                new_batches = max(1, processed_batches - resumed_batches)
                rate = new_batches / elapsed
                eta = (total_batches - processed_batches) / rate if rate else 0.0
                console.print(
                    f"Fact encoding: {batch_number}/{total_batches} batches "
                    f"({min(start + 128, len(facts))}/{len(facts)} facts, ETA {eta:.0f}s)"
                )

    if memory_map is not None:
        del memory_map
    if partial_path.exists():
        os.replace(partial_path, final_embeddings_path)
    write_json(directory / "ids.json", [fact.fact_id for fact in facts])
    manifest = {
        "retriever_hash": state_dict_hash(model),
        "retriever_artifact_signature": artifact_signature,
        "data_hash": index.data_hash,
        "kg_data_hash": kg_manifest["data_hash"],
        "config_hash": cfg.hash,
        "facts": len(facts),
        "limit": limit,
    }
    write_json(directory / "manifest.json", manifest)
    partial_manifest_path.unlink(missing_ok=True)
    console.print(f"Encoded {len(facts)} facts")


@app.command("build-fact-index")
def build_fact_index(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
                     log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Build an exact cosine fact index from current embeddings."""
    cfg, _ = _context(config, seed, log_level, set_values)
    source = cfg.paths.cache_root / "fact_embeddings"
    manifest = read_json(source / "manifest.json")
    target = cfg.paths.cache_root / "fact_index"
    source_embeddings = source / "embeddings.npy"
    target_embeddings = target / "embeddings.npy"

    def install_shared_embeddings() -> None:
        """Use one inode for embeddings instead of storing a second copy."""
        target.mkdir(parents=True, exist_ok=True)
        if (
            target_embeddings.exists()
            and os.path.samefile(source_embeddings, target_embeddings)
        ):
            return
        temporary = target / "embeddings.npy.link"
        temporary.unlink(missing_ok=True)
        os.link(source_embeddings, temporary)
        os.replace(temporary, target_embeddings)

    if resume != "none" and all((target / name).exists() for name in ("embeddings.npy", "ids.json", "manifest.json")):
        existing = read_json(target / "manifest.json")
        if existing == manifest:
            install_shared_embeddings()
            console.print(f"Fact-index cache hit: reusing {existing['facts']} vectors in {target}")
            return
    ids = read_json(source / "ids.json")
    install_shared_embeddings()
    write_json(target / "ids.json", ids)
    write_json(target / "manifest.json", manifest)
    console.print(f"Fact index ready with {len(ids)} vectors (shared embedding storage)")


@app.command("build-reranker-data")
def build_reranker_data(config: ConfigPath, retriever_checkpoint: str = "best", resume: str = "none",
                        limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
                        set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Convert mined retriever triples to labeled cross-encoder pairs."""
    cfg, _ = _context(config, seed, log_level, set_values)
    output = cfg.paths.cache_root / "reranker_pairs" / "train.jsonl"
    source_path = cfg.paths.cache_root / "retrieval_pairs" / "train.jsonl"
    source_manifest = read_json(source_path.with_suffix(".manifest.json"))
    retriever_data_signature = object_hash({
        key: value
        for key, value in source_manifest.items()
        if key not in {"resolved_config_hash", "config_hash"}
    })
    expected_manifest = {
        "reranker_data_version": 1,
        "retriever_data_signature": retriever_data_signature,
        "limit": limit,
    }
    manifest_path = output.with_suffix(".manifest.json")
    if resume != "none" and output.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected_manifest.items()):
            console.print(f"Reranker-data cache hit: reusing {manifest['records']} pairs in {output}")
            return
    source = _read_jsonl(source_path)
    records = build_reranker_records(source[:limit] if limit else source)
    write_jsonl(output, records)
    write_json(manifest_path, {
        **expected_manifest,
        "records": len(records),
        "resolved_config_hash": cfg.hash,
    })
    console.print(f"Wrote {len(records)} reranker pairs")


@app.command("train-reranker")
def train_reranker(config: ConfigPath, resume: str = "auto", limit: int | None = None, seed: int | None = None,
                   log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Fine-tune the complete local BGE reranker and verify updates."""
    cfg, raw = _context(config, seed, log_level, set_values)
    run_dir = _run_dir(cfg, "reranker")
    completed = (
        completed_transformer_training(run_dir)
        if resume != "none"
        else None
    )
    if completed is not None:
        console.print(f"Reranker-training cache hit: reusing {run_dir}")
        console.print_json(data=completed)
        return
    records = _read_jsonl(cfg.paths.cache_root / "reranker_pairs" / "train.jsonl")
    model = BGEReranker(cfg.paths.pretrained_reranker)
    result = fit_reranker(model, records[:limit] if limit else records, run_dir,
                          cfg.model_extra["reranker_training"], select_device(), resume)
    if cfg.model_extra["checkpoint"]["cleanup_after_successful_save"]:
        cleanup_run(run_dir / "checkpoints",
                    int(cfg.model_extra["checkpoint"]["keep_last"]), execute=True)
    save_resolved_config(cfg, raw, run_dir)
    console.print_json(data=result)


def _chain_cache(config: Config, split: str, limit: int | None, reuse: bool = True) -> tuple[list[dict], list[dict]]:
    index = _kg_index(config)
    component_variant = candidate_generation_variant(config)
    fact_index_dir = config.paths.cache_root / "fact_index"
    if (
        component_variant != NO_DENSE_RETRIEVER
        and not (fact_index_dir / "manifest.json").exists()
    ):
        raise RuntimeError("Fact index missing; run decgc encode-facts and decgc build-fact-index first")
    fact_index = (
        None
        if component_variant == NO_DENSE_RETRIEVER
        else FactIndex.load(fact_index_dir)
    )
    retriever_path = _best_model(config, "retriever", config.paths.pretrained_retriever)
    reranker_path = _best_model(config, "reranker", config.paths.pretrained_reranker)
    # Derive the summary from the model path actually selected. This preserves
    # candidate-cache validity when a named four-loss experiment reuses the
    # legacy standalone reranker checkpoint.
    reranker_summary = reranker_path.parent / "training_summary.json"
    reranker_hash = read_json(reranker_summary)["after_hash"] if reranker_summary.exists() else object_hash(str(reranker_path))
    examples = _examples(config, split, limit)
    _ensure_local_parse_cache(config, split, examples)
    settings = config.model_extra["retrieval"]
    manifest = {
        "chain_cache_version": 6,
        "retriever_hash": (
            AnswerBlindLexicalEncoder.version
            if component_variant == NO_DENSE_RETRIEVER
            else fact_index.manifest["retriever_hash"]
        ),
        "reranker_hash": (
            "disabled-candidate-reranker"
            if component_variant == NO_CROSS_ENCODER_RERANKER
            else reranker_hash
        ),
        "data_hash": index.data_hash,
        "fact_index_signature": (
            "not-used"
            if fact_index is None
            else object_hash(fact_index.manifest)
        ),
        "parse_source": file_signature(_parse_path(config, split)),
        "search_signature": object_hash({
            "retrieval": settings,
            "stop_threshold": config.model_extra["model"]["stop_threshold"],
            "prune_threshold": config.model_extra["model"]["prune_threshold"],
        }),
        "split": split,
    }
    if component_variant != "baseline":
        manifest["component_ablation"] = component_variant
    chain_cache_root = candidate_chain_root(config)
    cache_dir = chain_cache_root / split
    parses = _parsed(config, split)
    all_completed, all_audit, missing = [], [], []
    for example in examples:
        shard = chain_shard_path(cache_dir, example.qid)
        if reuse and chain_shard_is_current(
            chain_cache_root, split, example.qid, manifest,
        ):
            rows = _read_jsonl(shard)
            all_audit.extend(rows)
            all_completed.extend(row for row in rows if row.get("state") == "COMPLETED")
        else:
            missing.append(example)
    if not missing:
        return all_completed, all_audit

    retriever = (
        AnswerBlindLexicalEncoder().to(select_device()).eval()
        if component_variant == NO_DENSE_RETRIEVER
        else BGEEncoder(retriever_path).to(select_device()).eval()
    )
    reranker = (
        None
        if component_variant == NO_CROSS_ENCODER_RERANKER
        else BGEReranker(reranker_path).to(select_device()).eval()
    )
    search = TemporalBeamSearch(
        index,
        retriever=retriever,
        reranker=reranker,
        fact_index=fact_index,
        beam_size=settings["beam_size"],
        max_hops=settings["max_hops"],
        retrieve_top_k=settings["retrieve_top_k"],
        rerank_top_k=settings["rerank_top_k"],
        rerank_batch_size=min(16, settings["rerank_top_k"]),
        stop_threshold=config.model_extra["model"]["stop_threshold"],
        prune_threshold=config.model_extra["model"]["prune_threshold"],
        max_paths_per_answer=int(settings.get("max_paths_per_answer", 2)),
    )
    chain_started = time.monotonic()
    total_missing = len(missing)
    for processed, example in enumerate(missing, 1):
        completed, audit = search.run(example.question, parses[example.qid])
        write_chain_cache(chain_cache_root, split, example.qid, audit, manifest)
        all_completed.extend(chain.model_dump(mode="json") for chain in completed)
        all_audit.extend(chain.model_dump(mode="json") for chain in audit)
        if processed == total_missing or processed % 100 == 0:
            elapsed = max(time.monotonic() - chain_started, 1e-6)
            rate = processed / elapsed
            eta = (total_missing - processed) / rate if rate else 0.0
            console.print(
                f"Chain cache ({split}): {processed}/{total_missing} questions "
                f"({rate:.2f} questions/s, ETA {eta:.0f}s)"
            )
    return all_completed, all_audit


def _load_reasoner(config: Config):
    checkpoint_root = _reasoner_checkpoint_root(config)
    manager = CheckpointManager(checkpoint_root)
    checkpoint = manager.resolve("best")
    if checkpoint is None:
        return None
    retriever = BGEEncoder(_best_model(config, "retriever", config.paths.pretrained_retriever))
    reranker = BGEReranker(_best_model(config, "reranker", config.paths.pretrained_reranker))
    model = DECGCModel(32, config.model_extra["model"]["hidden_size"], retriever, reranker).to(select_device())
    explicit_reasoner = config.model_extra.get("ablation_study", {}).get(
        "reasoner_run_dir"
    )
    manager.load(
        checkpoint,
        model,
        expected_config_hash=(
            None
            if explicit_reasoner is not None
            else stage_config_hash(config, "reasoner")
        ),
    )
    return model.eval()


def _rescore_with_reasoner(
    model,
    question: str,
    chains: list[ChainPrefix],
    query: StructuredQuery,
) -> None:
    if model is None or not chains:
        return
    reliability_logits = []
    stop_logits = []
    chain_states = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for chain in chains:
            previous = torch.zeros(1, model.chain_encoder.gru.hidden_size, device=device)
            history = []
            output = None
            previous_reliability = 0.0
            for hop, fact in enumerate(chain.facts, 1):
                encoded = model.encode_live([question], [format_fact(fact)])
                current_type_score = answer_type_score(
                    chain.candidate_answer,
                    query.answer_type,
                )
                output = model.prefix_step(
                    previous, encoded, torch.tensor([hop], device=device),
                    torch.tensor([chain.temporal_score], device=device),
                    torch.tensor([current_type_score], device=device),
                )
                previous = output["chain"]
                current_reliability = float(output["absolute_reliability"][0])
                history.append({
                    "hop": hop, "absolute_reliability": current_reliability,
                    "relative_weight": 0.0, "stop_probability": float(output["stop_probability"][0]),
                    "state": chain.state,
                    "evidence_gain": current_reliability - previous_reliability,
                })
                previous_reliability = current_reliability
            if output is None:
                reliability_logits.append(torch.tensor(float("-inf"), device=device))
                stop_logits.append(torch.tensor(float("-inf"), device=device))
                chain_states.append(torch.zeros(
                    model.chain_encoder.gru.hidden_size,
                    device=device,
                ))
                continue
            chain.absolute_reliability = float(output["absolute_reliability"][0])
            chain.stop_probability = float(output["stop_probability"][0])
            chain.type_score = answer_type_score(
                chain.candidate_answer,
                query.answer_type,
            )
            chain.state_history = [ChainSnapshot.model_validate(item) for item in history]
            reliability_logits.append(output["reliability_logits"][0])
            stop_logits.append(output["stop_logits"][0])
            chain_states.append(output["chain"][0])
        stacked_states = torch.stack(chain_states)
        answer_texts = [chain.candidate_answer or "" for chain in chains]
        group_ids = [chain.qid for chain in chains]
        answer_logits = model.score_answers(stacked_states, answer_texts)
        interaction_logits = model.score_interactions(
            stacked_states,
            answer_texts,
            group_ids,
        )
        interaction_confidences = torch.sigmoid(interaction_logits).cpu().tolist()
        normalized_answers = [normalize_answer(answer) for answer in answer_texts]
        for index, chain in enumerate(chains):
            interactions = []
            for other_index, other in enumerate(chains):
                if index == other_index or chain.qid != other.qid:
                    continue
                same_answer = normalized_answers[index] == normalized_answers[other_index]
                interactions.append({
                    "other_chain_id": other.chain_id,
                    "interaction_type": "cooperation" if same_answer else "competition",
                    "interaction_reason": (
                        "same normalized candidate answer"
                        if same_answer
                        else "different normalized candidate answer"
                    ),
                    "interaction_confidence": float(interaction_confidences[index]),
                })
            chain.graph_interactions = interactions
        combined_logits = (
            torch.stack(reliability_logits)
            + torch.stack(stop_logits)
            + answer_logits
            + interaction_logits
        )
        answer_probabilities = torch.sigmoid(
            answer_logits + interaction_logits
        ).cpu().tolist()
        answer_groups: dict[str, list[int]] = {}
        for index, answer in enumerate(answer_texts):
            answer_groups.setdefault(normalize_answer(answer), []).append(index)
        group_keys = list(answer_groups)
        group_logits = torch.stack([
            combined_logits[answer_groups[key]].max()
            for key in group_keys
        ])
        group_weights = torch.softmax(group_logits, dim=0)
        weights_by_answer = {
            key: float(weight)
            for key, weight in zip(group_keys, group_weights.cpu())
        }
        for chain, answer_probability in zip(
            chains,
            answer_probabilities,
        ):
            chain.answer_score = float(answer_probability)
            chain.relative_weight = weights_by_answer[
                normalize_answer(chain.candidate_answer or "")
            ]
            for snapshot in chain.state_history:
                snapshot.relative_weight = chain.relative_weight


@app.command("build-chain-cache")
def build_chain_cache(config: ConfigPath, split: str = "train", retriever_checkpoint: str = "best",
                      reranker_checkpoint: str = "best", resume: str = "none", limit: int | None = None,
                      seed: int | None = None, log_level: str = "INFO",
                      set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Generate sharded candidate chains without consulting gold answers."""
    cfg, _ = _context(config, seed, log_level, set_values)
    completed, audit = _chain_cache(cfg, split, limit, reuse=resume != "none")
    console.print(f"Cached {len(audit)} prefixes ({len(completed)} completed)")


@app.command("build-counterfactuals")
def build_counterfactuals(config: ConfigPath, split: str = "train", resume: str = "none",
                          limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
                          set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Validate on-demand counterfactual inputs without materializing duplicates."""
    cfg, _ = _context(config, seed, log_level, set_values)
    output = cfg.paths.cache_root / "counterfactuals" / f"{split}.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    # Version 3 generates the single counterfactual used by each reasoner
    # sample on demand. The old four-copies-per-chain JSONL and any failed
    # atomic temporary are therefore safe to remove before doing more work.
    legacy_cleanup = remove_scoped_paths(
        [output, output.with_suffix(output.suffix + ".tmp")],
        cfg.paths.outputs_root,
        execute=True,
    )
    if legacy_cleanup["removed"]:
        console.print(
            f"Removed legacy materialized counterfactual data "
            f"({legacy_cleanup['bytes_to_release'] / 2**20:.1f} MiB)"
        )
    source = cfg.paths.cache_root / "candidate_chains" / split
    files = chain_shard_files(source)
    selected_files = files[:limit] if limit else files
    source_signature = object_hash([
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in selected_files
    ])
    expected_manifest = {
        "counterfactual_data_version": 3,
        "split": split,
        "limit": limit,
        "source_signature": source_signature,
        "materialized": False,
        "generation": "on_demand",
    }
    if resume != "none" and manifest_path.exists():
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected_manifest.items()):
            console.print(
                f"Counterfactual manifest hit: {manifest['records']} "
                "perturbations will be generated on demand"
            )
            return
    completed_chains = 0
    for path in selected_files:
        for payload in _read_jsonl(path):
            if payload.get("state") == "COMPLETED":
                completed_chains += 1
    write_json(manifest_path, {
        **expected_manifest,
        "completed_chains": completed_chains,
        "records": completed_chains * 4,
        "resolved_config_hash": cfg.hash,
    })
    console.print(
        f"Validated {completed_chains} completed chains; "
        f"{completed_chains * 4} counterfactual perturbations will be "
        "generated on demand without a duplicate JSONL"
    )


def _catalog_alternative(values: list[str], current: str, seed_value: int) -> str:
    if not values:
        return "__counterfactual__"
    for offset in range(len(values)):
        candidate = values[(seed_value + offset) % len(values)]
        if candidate != current:
            return candidate
    return "__counterfactual__"


def _build_reasoner_samples(
    cfg: Config,
    split: str,
    hidden: int,
    fact_size: int,
    max_per_question: int,
    limit: int | None = None,
) -> tuple[list[dict], int, int, int]:
    """Create weakly supervised, question-grouped prefix samples for one split."""
    chain_files = chain_shard_files(
        cfg.paths.cache_root / "candidate_chains" / split
    )
    examples = {example.qid: example for example in _examples(cfg, split)}
    if split == "train":
        entity_catalog, relation_catalog = _kg_catalog(cfg)
        entity_values = sorted(entity_catalog)
        relation_values = sorted(relation_catalog)
    else:
        entity_values, relation_values = [], []
    samples = []
    positive_samples = 0
    questions_with_positive: set[str] = set()
    used_questions = 0
    for path in chain_files:
        candidate_rows = _read_jsonl(path)
        if not candidate_rows:
            continue
        qid = str(candidate_rows[0]["qid"])
        example = examples.get(qid)
        if example is None:
            continue
        if limit is not None and used_questions >= limit:
            break
        used_questions += 1
        for chain, weak_positive in select_balanced_prefixes(
            candidate_rows,
            example.answers,
            max_per_question,
            example.answer_type,
        ):
            if weak_positive:
                positive_samples += 1
                questions_with_positive.add(example.qid)
            fact_text = format_fact(chain.facts[-1])
            counterfactual_texts: list[str] = []
            counterfactual_types: list[str] = []
            if split == "train" and weak_positive:
                seed_value = int(object_hash(chain.chain_id)[:8], 16)
                counterfactuals = make_counterfactuals(
                    chain,
                    _catalog_alternative(
                        relation_values,
                        chain.facts[-1].relation_id,
                        seed_value,
                    ),
                    _catalog_alternative(
                        entity_values,
                        chain.candidate_answer or chain.current_entity,
                        seed_value,
                    ),
                )
                for payload in counterfactuals:
                    counterfactual = ChainPrefix.model_validate(payload)
                    counterfactual_texts.append(format_fact(counterfactual.facts[-1]))
                    counterfactual_types.append(str(payload["counterfactual_type"]))
            current_type_score = answer_type_score(
                chain.candidate_answer,
                example.answer_type,
            )
            stop_positive = is_optimal_stop_prefix(
                chain,
                candidate_rows,
                example.answers,
                example.answer_type,
            )
            samples.append({
                "qid": example.qid,
                "previous": torch.zeros(1, hidden),
                "fact": torch.zeros(1, fact_size),
                "hop": torch.tensor([chain.hop]),
                "temporal": torch.tensor([chain.temporal_score], dtype=torch.float),
                "type_score": torch.tensor([current_type_score], dtype=torch.float),
                "prefix_label": torch.tensor([1.0 if weak_positive else 0.0]),
                "stop_label": torch.tensor([1.0 if stop_positive else 0.0]),
                "questions": [example.question],
                "fact_texts": [fact_text],
                "answer_texts": [chain.candidate_answer],
                "fact_sequences": [[format_fact(fact) for fact in chain.facts]],
                "cf_fact_texts": counterfactual_texts,
                "cf_types": counterfactual_types,
                "cf_mask": torch.tensor([1.0 if split == "train" and weak_positive else 0.0]),
            })
    return (
        samples,
        positive_samples,
        len(questions_with_positive),
        used_questions,
    )


@app.command("train-reasoner")
def train_reasoner(config: ConfigPath, retriever_checkpoint: str = "best", reranker_checkpoint: str = "best",
                   resume: str = "auto", limit: int | None = None, seed: int | None = None,
                   log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Train reliability/stop dynamics from cached weak labels and counterfactuals."""
    cfg, raw = _context(config, seed, log_level, set_values)
    completed_summary = _run_dir(cfg, "reasoner") / "training_summary.json"
    if resume != "none" and completed_summary.exists():
        summary = read_json(completed_summary)
        if summary.get("training_complete"):
            console.print(f"Reasoner-training cache hit: reusing {_run_dir(cfg, 'reasoner')}")
            console.print_json(data=summary)
            return
    hidden, fact_size = cfg.model_extra["model"]["hidden_size"], 32
    retriever = BGEEncoder(_best_model(cfg, "retriever", cfg.paths.pretrained_retriever))
    retriever.train_last_n_layers(int(cfg.model_extra["model"]["trainable_last_n_layers"]))
    reranker = BGEReranker(_best_model(cfg, "reranker", cfg.paths.pretrained_reranker))
    if cfg.model_extra["reasoner_training"].get("gradient_checkpointing"):
        retriever.model.gradient_checkpointing_enable()
        reranker.model.gradient_checkpointing_enable()
    for parameter in reranker.parameters():
        parameter.requires_grad = True
    model = DECGCModel(fact_size, hidden, retriever, reranker).to(select_device())
    optimizer = build_optimizer(model, retriever, reranker, cfg.model_extra["reasoner_training"])
    max_per_question = int(cfg.model_extra["reasoner_training"].get("max_prefixes_per_question", 2))
    batch_size = int(cfg.model_extra["reasoner_training"].get("batch_size", 1))
    samples, positive_samples, positive_questions, used_questions = _build_reasoner_samples(
        cfg,
        "train",
        hidden,
        fact_size,
        max_per_question,
        limit,
    )
    batches = collate_question_batches(samples, batch_size)
    validation_limit = int(
        cfg.model_extra["reasoner_training"].get("validation_max_questions", 500)
    )
    validation_samples, validation_positives, validation_positive_questions, validation_questions = (
        _build_reasoner_samples(
            cfg,
            "valid",
            hidden,
            fact_size,
            max_per_question,
            validation_limit,
        )
    )
    validation_batches = collate_question_batches(validation_samples, batch_size)
    console.print(
        f"Reasoner training set: {len(samples)} representative prefixes "
        f"from {used_questions} questions in {len(batches)} batches; "
        f"{positive_samples} weak positives covering {positive_questions} questions"
    )
    console.print(
        f"Reasoner validation set: {len(validation_samples)} representative prefixes "
        f"from {validation_questions} questions in {len(validation_batches)} batches; "
        f"{validation_positives} weak positives covering "
        f"{validation_positive_questions} questions"
    )
    training_positive_coverage = positive_questions / max(1, used_questions)
    validation_positive_coverage = (
        validation_positive_questions / max(1, validation_questions)
    )
    minimum_positive_coverage = float(
        cfg.model_extra.get("quality_gate", {}).get(
            "reasoner_positive_question_coverage_min",
            0.0,
        )
    )
    if (
        training_positive_coverage < minimum_positive_coverage
        or validation_positive_coverage < minimum_positive_coverage
    ):
        message = (
            "Reasoner candidate-positive coverage is insufficient: "
            f"train={training_positive_coverage:.3%}, "
            f"valid={validation_positive_coverage:.3%}, "
            f"required={minimum_positive_coverage:.3%}"
        )
        if bool(
            cfg.model_extra.get("quality_gate", {}).get(
                "blocking",
                True,
            )
        ):
            raise RuntimeError(message)
        console.print(
            f"[yellow]{message}; continuing because "
            "quality_gate.blocking=false.[/yellow]"
        )
    if positive_samples == 0:
        raise RuntimeError(
            "Reasoner training has no weak-positive prefixes; candidate ranking cannot be learned"
        )
    result = fit_reasoner(
        model,
        batches,
        optimizer,
        cfg,
        _run_dir(cfg, "reasoner"),
        resume,
        validation_batches=validation_batches,
        checkpoint_config_hash=stage_config_hash(cfg, "reasoner"),
    )
    if cfg.model_extra["checkpoint"]["cleanup_after_successful_save"]:
        cleanup_run(_run_dir(cfg, "reasoner") / "checkpoints",
                    int(cfg.model_extra["checkpoint"]["keep_last"]), execute=True)
    save_resolved_config(cfg, raw, _run_dir(cfg, "reasoner"))
    console.print_json(data=result)


def _evaluate_question(
    config: Config,
    example,
    cached_query: StructuredQuery,
    evaluation_parser: LocalQuestionParser,
    index: KGIndex,
    reasoner,
    completed_rows: list[dict],
    audit_rows: list[dict],
    export_graph: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Evaluate one question while retaining only that question's chains."""
    completed = [ChainPrefix.model_validate(row) for row in completed_rows]
    audited = [ChainPrefix.model_validate(row) for row in audit_rows]
    local_query = evaluation_parser.parse(
        example.qid,
        example.question,
        example.answer_type,
        {
            key: value
            for key, value in example.raw_record.items()
            if key != "answers"
        },
    )
    query = cached_query.model_copy(update={
        "topic_entities": local_query.topic_entities or cached_query.topic_entities,
        "target_entities": local_query.target_entities or cached_query.target_entities,
        "relation_hints": local_query.relation_hints or cached_query.relation_hints,
        "temporal_constraints": (
            local_query.temporal_constraints or cached_query.temporal_constraints
        ),
        "temporal_operator": (
            local_query.temporal_operator or cached_query.temporal_operator
        ),
        "temporal_granularity": (
            local_query.temporal_granularity or cached_query.temporal_granularity
        ),
        "answer_type": local_query.answer_type or cached_query.answer_type,
        "answer_role": local_query.answer_role or cached_query.answer_role,
    })
    chains = stop_candidates(audited + completed)
    _rescore_with_reasoner(reasoner, example.question, chains, query)
    neural_ranking, scores = aggregate_completed(chains)
    symbolic = direct_temporal_candidates(
        index,
        example.question,
        query,
    )
    ranking = merge_symbolic_answers(neural_ranking, symbolic)
    existing_symbolic_keys = {
        (
            chain.facts[-1].fact_id,
            normalize_answer(chain.candidate_answer or ""),
        )
        for chain in chains
        if chain.facts
    }
    output_audit = list(audit_rows)
    for answer, fact in zip(symbolic.answers, symbolic.facts):
        key = (fact.fact_id, normalize_answer(answer))
        if key in existing_symbolic_keys:
            continue
        symbolic_chain = ChainPrefix(
            chain_id=object_hash([
                "symbolic-direct-v1",
                example.qid,
                fact.fact_id,
                normalize_answer(answer),
            ])[:20],
            qid=example.qid,
            hop=1,
            facts=[fact],
            entity_path=[fact.subject_id, fact.object_id],
            relation_path=[fact.relation_id],
            time_path=[fact.timestamp_raw],
            current_entity=(
                fact.subject_id
                if query.answer_role == "subject"
                else fact.object_id
            ),
            candidate_answer=answer,
            retrieval_scores=[1.0],
            reranker_scores=[1.0],
            semantic_score=1.0,
            temporal_score=1.0,
            type_score=1.0,
            legality_score=1.0,
            absolute_reliability=1.0,
            relative_weight=1.0,
            stop_probability=1.0,
            answer_score=1.0,
            state="COMPLETED",
            stop_reason=f"symbolic:{symbolic.rule}",
        )
        chains.append(symbolic_chain)
        output_audit.append(symbolic_chain.model_dump(mode="json"))
        existing_symbolic_keys.add(key)
        scores.setdefault(answer, 1.0 if symbolic.prefer_symbolic else 0.0)

    candidate_chain_counts: Counter[str] = Counter()
    candidate_best_hops: dict[str, int] = {}
    for chain in chains:
        if not chain.candidate_answer:
            continue
        answer_key = normalize_answer(chain.candidate_answer)
        candidate_chain_counts[answer_key] += 1
        candidate_best_hops[answer_key] = min(
            candidate_best_hops.get(answer_key, chain.hop),
            chain.hop,
        )
    fallback = not bool(ranking)
    if fallback:
        ranking = [
            audited[0].candidate_answer
        ] if audited and audited[0].candidate_answer else []
    predicted = ranking[0] if ranking else ""
    chains.sort(
        key=lambda chain: (
            normalize_answer(chain.candidate_answer or "")
            != normalize_answer(predicted),
            -float(chain.relative_weight),
        )
    )
    prediction = {
        "qid": example.qid,
        "question": example.question,
        "predicted_answer": predicted,
        "candidate_answers": ranking,
        "candidate_scores": scores,
        "gold_answers": example.answers,
        "question_type": example.question_type,
        "answer_type": example.answer_type,
        "temporal_operator": query.temporal_operator,
        "temporal_granularity": query.temporal_granularity,
        "answer_role": query.answer_role,
        "parse_source": cached_query.parse_source,
        "topic_entities": query.topic_entities,
        "target_entities": query.target_entities,
        "relation_hint_count": len(query.relation_hints),
        "neural_candidate_answers": neural_ranking,
        "symbolic_candidates": symbolic.answers,
        "symbolic_rule": symbolic.rule,
        "symbolic_preferred": symbolic.prefer_symbolic,
        "candidate_chain_counts": dict(candidate_chain_counts),
        "candidate_best_hops": candidate_best_hops,
        "fallback": fallback,
    }
    evidence, validations = [], []
    for chain in chains[:config.model_extra["evidence"]["top_supporting_chains"]]:
        validation = validate_chain(
            chain,
            predicted,
            index,
            query,
            completed_only=True,
        )
        validations.append(validation)
        if validation["eligible_supporting_chain"]:
            evidence.append(chain_evidence(chain, predicted, validation))
    if export_graph:
        graph_dir = (
            config.paths.demo_root
            / f"multitq-{config.hash[:12]}"
            / "graph_json"
        )
        graph = DynamicGraphBuilder().build(
            example.qid,
            max((chain.hop for chain in chains), default=0),
            chains,
        )
        write_json(graph_dir / f"{example.qid}.json", graph.to_dict())
    return prediction, evidence, validations, output_audit


def _evaluate(config: Config, split: str, limit: int | None, export_graph: bool = False) -> tuple[list, list, list, dict]:
    completed_rows, audit_rows = _chain_cache(config, split, limit)
    completed_by_qid: dict[str, list[ChainPrefix]] = {}
    audit_by_qid: dict[str, list[ChainPrefix]] = {}
    for row in completed_rows:
        chain = ChainPrefix.model_validate(row)
        completed_by_qid.setdefault(chain.qid, []).append(chain)
    for row in audit_rows:
        chain = ChainPrefix.model_validate(row)
        audit_by_qid.setdefault(chain.qid, []).append(chain)
    examples = _examples(config, split, limit)
    queries = _parsed(config, split)
    index = _kg_index(config)
    entity_catalog = set(index.adjacency)
    relation_catalog = {fact.relation_id for fact in index.facts.values()}
    evaluation_parser = LocalQuestionParser(
        EntityIndex(list(entity_catalog)),
        relation_catalog,
    )
    reasoner = _load_reasoner(config)
    predictions, evidence, validations = [], [], []
    graph_dir = config.paths.demo_root / f"multitq-{config.hash[:12]}" / "graph_json"
    for example in examples:
        cached_query = queries[str(example.qid)]
        local_query = evaluation_parser.parse(
            example.qid,
            example.question,
            example.answer_type,
            {
                key: value
                for key, value in example.raw_record.items()
                if key != "answers"
            },
        )
        # Runtime repair keeps the expensive persisted parse and chain caches
        # intact. Only non-empty deterministic fields replace stale parser
        # fields; API-only information remains available as fallback.
        query = cached_query.model_copy(update={
            "topic_entities": local_query.topic_entities or cached_query.topic_entities,
            "target_entities": local_query.target_entities or cached_query.target_entities,
            "relation_hints": local_query.relation_hints or cached_query.relation_hints,
            "temporal_constraints": (
                local_query.temporal_constraints
                or cached_query.temporal_constraints
            ),
            "temporal_operator": (
                local_query.temporal_operator
                or cached_query.temporal_operator
            ),
            "temporal_granularity": (
                local_query.temporal_granularity
                or cached_query.temporal_granularity
            ),
            "answer_type": local_query.answer_type or cached_query.answer_type,
            "answer_role": local_query.answer_role or cached_query.answer_role,
        })
        chains = stop_candidates(
            audit_by_qid.get(example.qid, []) + completed_by_qid.get(example.qid, [])
        )
        _rescore_with_reasoner(reasoner, example.question, chains, query)
        neural_ranking, scores = aggregate_completed(chains)
        symbolic = direct_temporal_candidates(
            index,
            example.question,
            query,
        )
        ranking = merge_symbolic_answers(neural_ranking, symbolic)
        existing_symbolic_keys = {
            (
                chain.facts[-1].fact_id,
                normalize_answer(chain.candidate_answer or ""),
            )
            for chain in chains
            if chain.facts
        }
        for answer, fact in zip(symbolic.answers, symbolic.facts):
            key = (fact.fact_id, normalize_answer(answer))
            if key in existing_symbolic_keys:
                continue
            symbolic_chain = ChainPrefix(
                chain_id=object_hash([
                    "symbolic-direct-v1",
                    example.qid,
                    fact.fact_id,
                    normalize_answer(answer),
                ])[:20],
                qid=example.qid,
                hop=1,
                facts=[fact],
                entity_path=[fact.subject_id, fact.object_id],
                relation_path=[fact.relation_id],
                time_path=[fact.timestamp_raw],
                current_entity=(
                    fact.subject_id
                    if query.answer_role == "subject"
                    else fact.object_id
                ),
                candidate_answer=answer,
                retrieval_scores=[1.0],
                reranker_scores=[1.0],
                semantic_score=1.0,
                temporal_score=1.0,
                type_score=1.0,
                legality_score=1.0,
                absolute_reliability=1.0,
                relative_weight=1.0,
                stop_probability=1.0,
                answer_score=1.0,
                state="COMPLETED",
                stop_reason=f"symbolic:{symbolic.rule}",
            )
            chains.append(symbolic_chain)
            audit_rows.append(symbolic_chain.model_dump(mode="json"))
            existing_symbolic_keys.add(key)
            scores.setdefault(answer, 1.0 if symbolic.prefer_symbolic else 0.0)
        candidate_chain_counts: Counter[str] = Counter()
        candidate_best_hops: dict[str, int] = {}
        for chain in chains:
            if not chain.candidate_answer:
                continue
            answer_key = normalize_answer(chain.candidate_answer)
            candidate_chain_counts[answer_key] += 1
            candidate_best_hops[answer_key] = min(
                candidate_best_hops.get(answer_key, chain.hop),
                chain.hop,
            )
        fallback = not bool(ranking)
        if fallback:
            audited = audit_by_qid.get(example.qid, [])
            ranking = [audited[0].candidate_answer] if audited and audited[0].candidate_answer else []
        predicted = ranking[0] if ranking else ""
        chains.sort(
            key=lambda chain: (
                normalize_answer(chain.candidate_answer or "")
                != normalize_answer(predicted),
                -float(chain.relative_weight),
            )
        )
        prediction = {"qid": example.qid, "question": example.question, "predicted_answer": predicted,
                      "candidate_answers": ranking, "candidate_scores": scores,
                      "gold_answers": example.answers, "question_type": example.question_type,
                      "answer_type": example.answer_type,
                      "temporal_operator": query.temporal_operator,
                      "temporal_granularity": query.temporal_granularity,
                      "answer_role": query.answer_role,
                      "parse_source": cached_query.parse_source,
                      "topic_entities": query.topic_entities,
                      "target_entities": query.target_entities,
                      "relation_hint_count": len(query.relation_hints),
                      "neural_candidate_answers": neural_ranking,
                      "symbolic_candidates": symbolic.answers,
                      "symbolic_rule": symbolic.rule,
                      "symbolic_preferred": symbolic.prefer_symbolic,
                      "candidate_chain_counts": dict(candidate_chain_counts),
                      "candidate_best_hops": candidate_best_hops,
                      "fallback": fallback}
        predictions.append(prediction)
        for chain in chains[:config.model_extra["evidence"]["top_supporting_chains"]]:
            validation = validate_chain(chain, predicted, index, query, completed_only=True)
            validations.append(validation)
            if validation["eligible_supporting_chain"]:
                evidence.append(chain_evidence(chain, predicted, validation))
        if export_graph:
            graph = DynamicGraphBuilder().build(example.qid, max((c.hop for c in chains), default=0), chains)
            write_json(graph_dir / f"{example.qid}.json", graph.to_dict())
    metrics = answer_metrics(predictions) | evidence_metrics(validations, predictions)
    return predictions, evidence, audit_rows, metrics


@app.command("demo-evaluate")
def demo_evaluate(config: ConfigPath, split: str = "valid", checkpoint: str = "best", resume: str = "none",
                  limit: int | None = 500, seed: int | None = None, log_level: str = "INFO",
                  set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Offline validation Demo; gold answers are used only after inference."""
    cfg, _ = _context(config, seed, log_level, set_values)
    run_id = f"multitq-{cfg.hash[:12]}"
    target = cfg.paths.demo_root / run_id
    required = ("demo_metrics.json", "demo_diagnostics.json", "demo_predictions.jsonl",
                 "demo_supporting_evidence_chains.jsonl", "demo_chain_audit.jsonl",
                 "demo_evidence_validation.json")
    evaluation_manifest_path = target / "demo_evaluation.manifest.json"
    expected_manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "config_hash": cfg.hash,
        "reasoner_stage_hash": stage_config_hash(cfg, "reasoner"),
        "split": split,
        "limit": limit,
    }
    if (
        resume != "none"
        and all((target / name).exists() for name in required)
        and evaluation_manifest_path.exists()
    ):
        existing = read_json(evaluation_manifest_path)
        if all(existing.get(key) == value for key, value in expected_manifest.items()):
            console.print(f"Evaluation cache hit: reusing {target}")
            return
    predictions, evidence, audit, metrics = _evaluate(cfg, split, limit)
    write_json(target / "demo_metrics.json", metrics)
    write_json(target / "demo_diagnostics.json", answer_diagnostics(predictions))
    write_jsonl(target / "demo_predictions.jsonl", predictions)
    write_jsonl(target / "demo_supporting_evidence_chains.jsonl", evidence)
    write_jsonl(target / "demo_chain_audit.jsonl", audit)
    write_json(target / "demo_evidence_validation.json", evidence_metrics(
        [item["validity"] for item in evidence], predictions))
    write_json(evaluation_manifest_path, expected_manifest)
    console.print_json(data=metrics)


@app.command("demo-predict")
def demo_predict(config: ConfigPath, split: str = "valid", checkpoint: str = "best",
                 num_examples: int = 20, export_evidence: bool = False, export_graph_json: bool = False,
                 export_human_readable: bool = False, resume: str = "none", limit: int | None = None,
                 seed: int | None = None, log_level: str = "INFO",
                 set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Export deterministic answer/evidence bundles and graph JSON."""
    cfg, _ = _context(config, seed, log_level, set_values)
    run_id = f"multitq-{cfg.hash[:12]}"
    prediction_dir = cfg.paths.predictions_root / run_id
    evidence_dir = cfg.paths.evidence_root / run_id
    prediction_manifest_path = prediction_dir / "prediction.manifest.json"
    expected_manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "config_hash": cfg.hash,
        "reasoner_stage_hash": stage_config_hash(cfg, "reasoner"),
        "split": split,
        "num_examples": num_examples,
        "export_evidence": export_evidence,
        "export_graph_json": export_graph_json,
        "export_human_readable": export_human_readable,
    }
    if (
        resume != "none"
        and prediction_dir.exists()
        and evidence_dir.exists()
        and prediction_manifest_path.exists()
    ):
        existing = read_json(prediction_manifest_path)
        if all(existing.get(key) == value for key, value in expected_manifest.items()):
            console.print(f"Prediction cache hit: reusing {prediction_dir} and {evidence_dir}")
            return
    predictions, evidence, audit, metrics = _evaluate(cfg, split, num_examples, export_graph_json)
    write_prediction_bundle(predictions, evidence, audit, metrics,
                            prediction_dir, evidence_dir)
    write_json(prediction_manifest_path, expected_manifest)
    console.print(f"Exported {len(predictions)} predictions and {len(evidence)} verified chains")


@app.command("generate-demo-report")
def generate_demo_report(config: ConfigPath, checkpoint: str = "best", resume: str = "none",
                         limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
                         set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Assemble the Demo metrics and case summary."""
    cfg, _ = _context(config, seed, log_level, set_values)
    target = cfg.paths.demo_root / f"multitq-{cfg.hash[:12]}"
    report_manifest_path = target / "demo_report.manifest.json"
    expected_manifest = {
        "report_version": 2,
        "metrics_hash": file_hash(target / "demo_metrics.json"),
    }
    if (
        resume != "none"
        and (target / "demo_report.md").exists()
        and report_manifest_path.exists()
    ):
        existing = read_json(report_manifest_path)
        if existing == expected_manifest:
            console.print(f"Report cache hit: reusing {target / 'demo_report.md'}")
            return
    metrics = read_json(target / "demo_metrics.json")
    predictions = _read_jsonl(target / "demo_predictions.jsonl")
    errors = error_analysis(predictions)
    by_id = {row["qid"]: row for row in predictions}
    successes = [by_id[qid] for qid in errors["correct"]]
    failures = [by_id[qid] for bucket in ("ranking_error", "retrieval_miss") for qid in errors[bucket]]
    summary = {"samples": len(predictions), **metrics,
               "note": "Validation Demo only; not a formal test-set result."}
    generate_report(target / "demo_report.md", summary, successes, failures)
    write_json(report_manifest_path, expected_manifest)
    console.print(f"Report: {target / 'demo_report.md'}")


@app.command("verify-demo-targets")
def verify_demo_targets(
    config: ConfigPath,
    seed: int | None = None,
    log_level: str = "INFO",
    set_values: list[str] = typer.Option(None, "--set"),
) -> None:
    """Report whether persisted Demo metrics meet acceptance targets."""
    cfg, _ = _context(config, seed, log_level, set_values)
    target = cfg.paths.demo_root / f"multitq-{cfg.hash[:12]}"
    metrics_path = target / "demo_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"Demo metrics are missing: {metrics_path}; run demo-evaluate first"
        )
    settings = cfg.model_extra.get("quality_gate", {})
    required = {
        "hits1": float(settings.get("hits1_min", 0.82)),
        "hits10": float(settings.get("hits10_min", 0.98)),
    }
    result = evaluate_metric_gate(read_json(metrics_path), required)
    blocking = bool(settings.get("blocking", True))
    result["blocking"] = blocking
    write_json(target / "demo_target_gate.json", result)
    console.print_json(data=result)
    if not result["passed"]:
        if blocking:
            raise typer.Exit(code=2)
        console.print(
            "[yellow]Demo metric targets were not met; results were preserved "
            "and the pipeline will finish because "
            "quality_gate.blocking=false.[/yellow]"
        )


@app.command("smoke-test")
def smoke_test(config: ConfigPath, resume: str = "none", limit: int | None = 4, seed: int | None = None,
               log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Run offline tiny inference over existing caches and verify traceability."""
    cfg, _ = _context(config, seed, log_level, set_values)
    predictions, evidence, _, metrics = _evaluate(cfg, "valid", limit)
    result = {"predictions": len(predictions), "evidence_chains": len(evidence), "metrics": metrics,
              "passed": bool(predictions) and all("qid" in row for row in predictions)}
    write_json(cfg.paths.demo_root / "smoke_test.json", result)
    console.print_json(data=result)
    if not result["passed"]:
        raise typer.Exit(1)


@app.command("cleanup-checkpoints")
def cleanup_checkpoints(config: ConfigPath, dry_run: bool = typer.Option(False, "--dry-run"),
                        execute: bool = typer.Option(False, "--execute"), resume: str = "none",
                        limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
                        set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Preview or safely remove unprotected old checkpoint files."""
    cfg, _ = _context(config, seed, log_level, set_values)
    if execute == dry_run:
        raise typer.BadParameter("Choose exactly one of --dry-run or --execute")
    results = {}
    for stage in ("retriever", "reranker", "reasoner"):
        directory = _run_dir(cfg, stage) / ("checkpoints" if stage == "reasoner" else "checkpoints")
        if directory.exists():
            results[stage] = cleanup_run(directory, int(cfg.model_extra["checkpoint"]["keep_last"]), execute)
    console.print_json(data=results)


@app.command("cleanup-obsolete-artifacts")
def cleanup_obsolete_artifacts(
    config: ConfigPath,
    execute: bool = typer.Option(False, "--execute"),
    resume: str = "none",
    limit: int | None = None,
    seed: int | None = None,
    log_level: str = "INFO",
    set_values: list[str] = typer.Option(None, "--set"),
) -> None:
    """Remove only artifacts that cannot be selected by the current resume."""
    cfg, _ = _context(config, seed, log_level, set_values)
    current_runs = {
        stage: _run_dir(cfg, stage)
        for stage in ("retriever", "reranker", "reasoner")
    }
    keep_old_runs = int(
        cfg.model_extra.get("checkpoint", {}).get("keep_old_runs", 1)
    )
    candidates = obsolete_artifact_candidates(
        cfg.paths.outputs_root,
        current_runs,
        keep_old_runs,
    )
    result = remove_scoped_paths(
        candidates,
        cfg.paths.outputs_root,
        execute=execute,
    )
    result["bytes_to_release_gib"] = round(
        result["bytes_to_release"] / 2**30,
        3,
    )
    console.print_json(data=result)


def _frozen_result_items(item_root: Path, examples):
    for example in examples:
        yield read_json_gzip(item_root / f"{example.qid}.json.gz")


def _frozen_human_readable_lines(item_root: Path, examples):
    for item in _frozen_result_items(item_root, examples):
        prediction = item["prediction"]
        yield f"# Question\n\n{prediction['question']}\n"
        yield f"## Predicted Answer\n\n{prediction['predicted_answer']}\n"
        chains = item["evidence"]
        if not chains:
            yield (
                "No completed supporting chain; this prediction is marked "
                "as fallback.\n"
            )
        for chain_index, chain in enumerate(chains, 1):
            yield f"## Supporting Chain {chain_index}\n"
            for hop, fact in enumerate(chain["facts"], 1):
                yield (
                    f"Hop {hop}: ({fact['subject_text']}, "
                    f"{fact['relation_text']}, {fact['object_text']}, "
                    f"{fact['timestamp_raw']})\n\n"
                    f"Source: {fact['source_relative_path']} row "
                    f"{fact['source_row_index']}\n"
                )
            yield (
                f"Reliability trajectory: "
                f"{chain['reliability_trajectory']}\n\n"
                f"Temporal validation: "
                f"{chain['validity']['temporal_constraints_satisfied']}\n"
            )


def _run_frozen_evaluation(
    cfg: Config,
    split: str,
    limit: int | None,
    resume: str,
) -> tuple[str, dict]:
    """Evaluate official data one question at a time with durable resume."""
    examples = _examples(cfg, split, limit)
    _ensure_local_parse_cache(cfg, split, examples)
    parse_path = _parse_path(cfg, split)
    parse_repair_path = _parse_repair_path(cfg, split)
    selection_hash = object_hash([example.qid for example in examples])
    chain_root = candidate_chain_root(cfg) / split
    chain_manifest_path = chain_root / "manifest.json"
    if not chain_manifest_path.exists():
        raise RuntimeError(
            f"Candidate-chain cache for {split} is missing; "
            "run decgc build-chain-cache first"
        )
    checkpoint_root = _reasoner_checkpoint_root(cfg)
    reasoner_checkpoint = CheckpointManager(checkpoint_root).resolve("best")
    signature = {
        "evaluation_version": EVALUATION_VERSION,
        "config_hash": cfg.hash,
        "split": split,
        "questions": len(examples),
        "selection_hash": selection_hash,
        "parse_source": (
            file_signature(parse_path) if parse_path.exists() else None
        ),
        "parse_repair_source": (
            file_signature(parse_repair_path)
            if parse_repair_path.exists()
            else None
        ),
        "chain_manifest": file_signature(chain_manifest_path),
        "reasoner_checkpoint": (
            file_signature(reasoner_checkpoint)
            if reasoner_checkpoint is not None
            else None
        ),
    }
    run_id = f"frozen-{object_hash(signature)[:12]}"
    prediction_dir = cfg.paths.predictions_root / run_id
    evidence_dir = cfg.paths.evidence_root / run_id
    item_root = prediction_dir / "resume_items"
    item_root.mkdir(parents=True, exist_ok=True)
    completion_path = prediction_dir / "evaluation_manifest.json"
    if resume != "none" and completion_path.exists():
        completion = read_json(completion_path)
        if (
            completion.get("signature") == signature
            and completion.get("completed") is True
            and (prediction_dir / "predictions.jsonl").exists()
        ):
            console.print(
                f"Reusing completed frozen evaluation: {run_id} "
                f"({len(examples)} questions)"
            )
            return run_id, completion["metrics"]

    queries = _parsed(cfg, split)
    missing_query_ids = [
        str(example.qid) for example in examples
        if str(example.qid) not in queries
    ]
    if missing_query_ids:
        raise RuntimeError(
            "Question parse cache is still incomplete after local repair; "
            f"missing={len(missing_query_ids)}, first_qid={missing_query_ids[0]}"
        )
    index = _kg_index(cfg)
    evaluation_parser = LocalQuestionParser(
        EntityIndex(list(index.adjacency)),
        {fact.relation_id for fact in index.facts.values()},
    )
    reasoner = _load_reasoner(cfg)
    started = time.monotonic()
    reused = 0
    for processed, example in enumerate(examples, 1):
        result_path = item_root / f"{example.qid}.json.gz"
        if resume != "none" and result_path.exists():
            try:
                existing = read_json_gzip(result_path)
                if str(existing.get("qid")) == str(example.qid):
                    reused += 1
                else:
                    result_path.unlink()
            except (OSError, EOFError, ValueError, json.JSONDecodeError):
                # An interrupted/non-atomic legacy item is recomputed alone;
                # all other durable per-question results remain reusable.
                result_path.unlink(missing_ok=True)
        if not result_path.exists() or resume == "none":
            shard = chain_shard_path(chain_root, example.qid)
            if not shard.exists():
                raise RuntimeError(
                    f"Candidate-chain shard is missing for qid={example.qid}; "
                    "rerun decgc build-chain-cache --resume auto"
                )
            rows = _read_jsonl(shard)
            completed_rows = [
                row for row in rows if row.get("state") == "COMPLETED"
            ]
            prediction, evidence, validations, output_audit = _evaluate_question(
                cfg,
                example,
                queries[str(example.qid)],
                evaluation_parser,
                index,
                reasoner,
                completed_rows,
                rows,
            )
            write_json_gzip(result_path, {
                "qid": example.qid,
                "prediction": prediction,
                "evidence": evidence,
                "validations": validations,
                "neural_audit_records": len(rows),
                # Canonical neural audit rows remain in the chain cache. Store
                # only additional symbolic rows to avoid duplicating millions
                # of records during the resumable phase.
                "symbolic_audit": output_audit[len(rows):],
            })
        if processed == len(examples) or processed % 100 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = processed / elapsed
            eta = (len(examples) - processed) / rate if rate else 0.0
            console.print(
                f"Frozen evaluation ({split}): {processed}/{len(examples)} "
                f"questions, {reused} resumed "
                f"({rate:.2f} questions/s, ETA {eta:.0f}s)"
            )

    def predictions():
        for item in _frozen_result_items(item_root, examples):
            yield item["prediction"]

    def evidence_rows():
        for item in _frozen_result_items(item_root, examples):
            yield from item["evidence"]

    def symbolic_audit_rows():
        for item in _frozen_result_items(item_root, examples):
            yield from item["symbolic_audit"]

    diagnostics = answer_diagnostics_iter(predictions())
    answer_result = {
        key: diagnostics["overall"][key]
        for key in (
            "hits1",
            "hits10",
            "mrr",
            "candidate_answer_recall",
            "mean_candidate_count",
        )
    }
    validation_count = 0
    neural_audit_records = 0
    symbolic_audit_records = 0
    validation_sums = {
        "evidence_fact_validity": 0,
        "temporal_validity": 0,
        "answer_chain_consistency": 0,
        "completed_chain_precision": 0,
    }
    fallback_count = 0
    for item in _frozen_result_items(item_root, examples):
        fallback_count += bool(item["prediction"].get("fallback"))
        neural_audit_records += int(item.get("neural_audit_records", 0))
        symbolic_audit_records += len(item["symbolic_audit"])
        for validation in item["validations"]:
            validation_count += 1
            validation_sums["evidence_fact_validity"] += bool(
                validation["all_facts_exist_in_kg"]
            )
            validation_sums["temporal_validity"] += bool(
                validation["temporal_constraints_satisfied"]
            )
            validation_sums["answer_chain_consistency"] += bool(
                validation["answer_supported_by_chain"]
            )
            validation_sums["completed_chain_precision"] += bool(
                validation["eligible_supporting_chain"]
            )
    validation_denominator = max(1, validation_count)
    metrics = answer_result | {
        key: value / validation_denominator
        for key, value in validation_sums.items()
    } | {
        "fallback_rate": fallback_count / max(1, len(examples)),
    }

    write_jsonl(prediction_dir / "predictions.jsonl", predictions())
    write_jsonl(evidence_dir / "supporting_evidence_chains.jsonl", evidence_rows())
    write_jsonl(
        evidence_dir / "symbolic_chain_audit.jsonl",
        symbolic_audit_rows(),
    )
    write_json(evidence_dir / "all_chain_audit.manifest.json", {
        "canonical_neural_chain_shards": str(chain_root),
        "neural_audit_records": neural_audit_records,
        "symbolic_audit_file": "symbolic_chain_audit.jsonl",
        "symbolic_audit_records": symbolic_audit_records,
        "total_audit_records": (
            neural_audit_records + symbolic_audit_records
        ),
        "note": (
            "Neural audit rows are referenced in the canonical compressed "
            "candidate-chain shards instead of being duplicated."
        ),
    })
    write_json(evidence_dir / "evidence_validation.json", metrics)
    write_text_lines(
        evidence_dir / "human_readable_evidence.md",
        _frozen_human_readable_lines(item_root, examples),
    )
    write_json(cfg.paths.reports_root / f"{run_id}.json", metrics)
    write_json(
        cfg.paths.reports_root / f"{run_id}.diagnostics.json",
        diagnostics,
    )
    write_json(completion_path, {
        "completed": True,
        "signature": signature,
        "run_id": run_id,
        "questions": len(examples),
        "metrics": metrics,
    })
    return run_id, metrics


@app.command()
def evaluate(config: ConfigPath, split: str = "test", checkpoint: str = "best", resume: str = "none",
             limit: int | None = None, seed: int | None = None, log_level: str = "INFO",
             set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Run memory-bounded, resumable frozen offline evaluation."""
    cfg, _ = _context(config, seed, log_level, set_values)
    if checkpoint != "best":
        raise typer.BadParameter(
            "Frozen evaluation currently supports --checkpoint best only"
        )
    run_id, metrics = _run_frozen_evaluation(cfg, split, limit, resume)
    console.print(f"Frozen evaluation output: {run_id}")
    console.print_json(data=metrics)


@app.command()
def analyze(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
            log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Create deterministic error buckets for the latest Demo."""
    cfg, _ = _context(config, seed, log_level, set_values)
    target = cfg.paths.demo_root / f"multitq-{cfg.hash[:12]}"
    result = error_analysis(_read_jsonl(target / "demo_predictions.jsonl"))
    write_json(cfg.paths.reports_root / "demo_error_analysis.json", result)
    console.print_json(data={key: len(value) for key, value in result.items()})


@app.command("export-results")
def export_results(config: ConfigPath, resume: str = "none", limit: int | None = None, seed: int | None = None,
                   log_level: str = "INFO", set_values: list[str] = typer.Option(None, "--set")) -> None:
    """Write a checksummed export manifest without duplicating large checkpoints."""
    cfg, _ = _context(config, seed, log_level, set_values)
    files = []
    for root in (cfg.paths.demo_root, cfg.paths.predictions_root, cfg.paths.evidence_root, cfg.paths.reports_root):
        files.extend({"path": path.relative_to(cfg.project.root).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
                     for path in root.rglob("*") if path.is_file())
    write_json(cfg.paths.manifests_root / "results_export_manifest.json", {"config_hash": cfg.hash, "files": files})
    console.print(f"Manifested {len(files)} result files")


if __name__ == "__main__":
    app()
