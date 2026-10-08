# MultiTQ paper-aligned four-loss main experiment

This directory is isolated from `multitq_3loss_src` and shares only the
project-level data, pretrained models, caches, and output root.

The reasoner objective is:

`1.0 L_answer + 0.5 L_prefix + 0.3 L_stop + 0.3 L_counterfactual`.

The legacy aggregate ranking loss is disabled. Answer supervision jointly
trains the answer scorer and answer-centred chain interaction head. Stop labels
mark the earliest sufficient prefix on each route. Every weak-positive training
prefix uses all four distinct counterfactual types: time, relation, bridge/entity,
and answer-tail. Evaluation exports evidence-gain trajectories and explicit
cooperation/competition records.

Training artifacts use:

`outputs/runs/<stage>/main-4loss-multitq-10k/multitq-<hash>/`

Existing standalone retriever/reranker checkpoints in the legacy unlabeled
directory are reused; only the reasoner must be retrained.

From the project root:

```bash
bash multitq_src/scripts/run_main_experiment.sh
```

All long stages use `--resume auto`; rerun the same command after interruption.
