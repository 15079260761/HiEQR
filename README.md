# HiEQR

Code and local resources for reproducing the main experiments on MultiTQ and
CronQuestions. Run the commands below on Linux from the repository root.

## Layout

- `multitq_src/`: MultiTQ implementation and experiment scripts
- `cronquestions_src/`: CronQuestions implementation and experiment scripts
- `data/`: prepared datasets
- `models/`: local BGE retriever and reranker checkpoints

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r multitq_src/requirements.txt
```

If online question parsing is needed, create `.env` in the repository root:

```dotenv
API_KEY=your_api_key
```

Run the main experiments:

```bash
# MultiTQ
bash multitq_src/scripts/run_main_experiment.sh

# CronQuestions
bash cronquestions_src/scripts/run_main_experiment.sh
```

The scripts automatically locate the repository root and support resuming. MultiTQ
artifacts are written to `outputs/`; CronQuestions artifacts are written to
`cronquestions_outputs/`. Run the two experiments sequentially when using one GPU.

For dataset-specific details, see `multitq_src/README.md` and
`cronquestions_src/README_CRONQUESTIONS.md`.
