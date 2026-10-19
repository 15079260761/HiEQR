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

Run the paper-aligned three-loss main experiments from Equation 7:

```bash
# MultiTQ: train on 10k questions, then evaluate on the full test set
bash multitq_src/scripts/run_main_experiment.sh

# CronQuestions: train on 10k questions, then evaluate on the full test set
bash cronquestions_src/scripts/run_main_experiment.sh
```

The scripts automatically locate the repository root and support resuming. MultiTQ
artifacts are written to `outputs/`; CronQuestions artifacts are written to
`cronquestions_outputs/`. Run the two experiments sequentially when using one GPU.

This root README is the repository's single reproduction guide; implementation
details are kept alongside the executable code and configuration files.
