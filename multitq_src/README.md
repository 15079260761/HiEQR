# DECGC: Dynamic Evidence Chain Graph Construction

This project implements a staged, resumable MultiTQ experiment. It uses the local
`bge-base-en-v1.5` bi-encoder and `bge-reranker-base` cross-encoder, question-aware
temporal beam search, prefix reliability and stopping, chain cooperation/competition,
weight-modulated pure-PyTorch graph reasoning, and provenance-validated evidence export.

The runtime root is the repository root (`HiEQR`). Source code writes generated artifacts only
under `outputs/`; it never changes the original MultiTQ files or pretrained model folders.
The official test split is reserved for a single frozen full evaluation.

## Server environment

The RTX 2080 Ti supports FP16 but not native BF16. Use a project-local Python virtual
environment. The root `.env` name is already occupied by the required DeepSeek key file,
so the virtual-environment directory must be `.venv`, not `.env`.

```bash
cd /path/to/HiEQR

# Create and activate an isolated Python 3.12 environment.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install the image-matched official CUDA 13.0 build.
python -m pip install \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130

# Install DECGC and all runtime/development dependencies inside .venv.
python -m pip install -e ./multitq_src
python -m pip install -r multitq_src/requirements-dev.txt

# Verify that the wheel sees the RTX 2080 Ti.
python -c "import torch; print('torch=', torch.__version__); print('wheel CUDA=', torch.version.cuda); print('CUDA available=', torch.cuda.is_available()); print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

The CUDA toolkit reported by the image is not used to compile PyTorch here; the official
wheel carries its CUDA runtime. The NVIDIA driver must be compatible with CUDA 13.0.

Do not put the API key in code. Keep it only in the root `.env`:

```dotenv
DEEPSEEK_API_KEY=your_actual_key
```

## First validation

```bash
cd /path/to/HiEQR
source .venv/bin/activate
decgc doctor --config multitq_src/configs/experiments/multitq_demo.yaml
decgc inspect-data --config multitq_src/configs/experiments/multitq_demo.yaml
pytest -q multitq_src/tests
```

`doctor` loads both local models with `local_files_only=True`, runs an encoder and
reranker probe, checks CUDA/VRAM/disk, and writes `outputs/manifests/doctor_report.json`.
`inspect-data` recursively writes the file inventory, inferred schema, and statistics
manifests before preprocessing.

Before a long online parsing run, verify the `.env` key and non-thinking JSON response with
one safe request:

```bash
decgc probe-llm --config multitq_src/configs/experiments/multitq_demo.yaml
```

The probe prints only whether the key is present; it never prints the key value.

## Complete Demo

```bash
bash multitq_src/scripts/run_demo_pipeline.sh
```

The pipeline is idempotent. With `--resume auto`, the first run creates persistent
artifacts and later runs validate inexpensive source fingerprints before reusing them:

- `outputs/cache/kg_indices/`: the 461,329-fact adjacency index plus compact
  entity/relation catalogs;
- `outputs/cache/question_samples/`: deterministic 10,000-train and 2,000-valid
  samples, so later commands do not reread all 386,787 raw training questions;
- `outputs/cache/question_parses/`: validated local/DeepSeek structured parses;
- `outputs/cache/fact_embeddings/` and `outputs/cache/fact_index/`: reusable dense
  vectors and search index;
- `outputs/cache/candidate_chains/`: one resumable shard per question;
- `outputs/runs/`: retriever, reranker, and reasoner checkpoints;
- `outputs/demo/`, `outputs/predictions/`, `outputs/evidence/`, and
  `outputs/reports/`: evaluation artifacts;
- `outputs/logs/demo_pipeline_YYYYMMDD_HHMMSS.log`: stage-by-stage wall-clock time.

The Demo pipeline parses and evaluates only train/valid. The held-out test split is
reserved for the one-time frozen full evaluation.

### Fast evaluation after inference-only changes

Evaluation version 10 adds a gold-free symbolic temporal completion layer over the
persisted neural candidates. It repairs stale structured-query fields in memory and
uses direct KG relation direction plus `first`, `last`, `before`, `after`, and
same-period constraints. It does not rewrite question parses or candidate-chain
shards. After uploading an inference-only update, rerun only:

```bash
decgc demo-evaluate \
  --config multitq_src/configs/experiments/multitq_demo.yaml \
  --split valid --checkpoint best --limit 500 --resume auto
decgc generate-demo-report \
  --config multitq_src/configs/experiments/multitq_demo.yaml \
  --checkpoint best --resume auto
decgc verify-demo-targets \
  --config multitq_src/configs/experiments/multitq_demo.yaml
```

For a CPU-only regression against already persisted predictions and the real full KG:

```bash
python -B multitq_src/scripts/quick_replay_evaluate.py \
  --run-dir outputs/demo/multitq-93f88bd186a8
```

Gold answers are used by this regression only after candidate generation to compute
metrics; the prediction rule receives only the question structure and KG facts.

For environments where the key should not be used, parse locally and cache all results:

```bash
decgc parse-questions \
  --config multitq_src/configs/experiments/multitq_demo.yaml \
  --split all --no-online --resume auto
```

Each training stage records pre/post parameter hashes and gradient norms. After retriever
fine-tuning, `encode-facts` necessarily produces a new manifest-bound fact index. Cache
manifests bind the data, retriever, reranker, and resolved configuration hashes.

Expected primary outputs:

- `outputs/runs/{retriever,reranker,reasoner}/...`
- `outputs/predictions/<run_id>/predictions.jsonl`
- `outputs/evidence/<run_id>/supporting_evidence_chains.jsonl`
- `outputs/evidence/<run_id>/human_readable_evidence.md`
- `outputs/demo/<run_id>/demo_report.md`

Preview checkpoint cleanup before execution:

```bash
decgc cleanup-checkpoints --config multitq_src/configs/experiments/multitq_demo.yaml --dry-run
decgc cleanup-checkpoints --config multitq_src/configs/experiments/multitq_demo.yaml --execute
```

Run the official test only after freezing configuration and checkpoints:

```bash
bash multitq_src/scripts/run_full_pipeline.sh
```

`decgc evaluate` is memory-bounded and resumable. It reads one compressed
candidate-chain shard at a time and durably writes one result item per
question. Re-run the same command with `--resume auto` after an interruption;
completed questions are skipped:

```bash
decgc evaluate \
  --config multitq_src/configs/experiments/multitq_demo.yaml \
  --split test \
  --checkpoint best \
  --resume auto \
  --set demo.evaluation_max_samples=null \
  --set evaluation.split=test \
  --set evaluation.max_samples=null
```

The final `predictions.jsonl` and completion manifest are written only after
all selected questions have durable result items. The large neural chain audit
is referenced from the canonical compressed candidate-chain cache rather than
duplicated under the evaluation output.

Export the persisted frozen-test diagnostics as a paper-ready table. This
post-processing step does not load models or run inference:

```bash
python -B multitq_src/scripts/export_paper_metrics.py \
  --model DECGC \
  --decimals 1
```

It writes `frozen-*.paper_metrics.{json,csv,md}` under `outputs/reports/` with
Hits@1 and Hits@10 columns for Overall, Multiple, Single, Entity, and Time.

Training and evaluation never call DeepSeek. Only `parse-questions` can do so, and every
validated response is cached. Evidence prose is generated from deterministic templates;
the model never invents KG facts.

### DeepSeek empty JSON responses

DeepSeek documents that JSON mode can occasionally return an empty `content` even with
HTTP 200. The parser therefore retries the primary and fallback models, accepts plain or
Markdown-wrapped JSON, validates every allow-listed field, and falls back to the local
parser instead of aborting the pipeline. Split parsing is durably appended to a
`.partial.jsonl` file and atomically finalized, so interruption does not discard completed
questions. The resulting manifest reports counts for `local`, `deepseek`, and
`local_fallback`.
