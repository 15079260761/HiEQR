# MultiTQ paper-aligned three-loss main experiment

The implemented reasoner objective follows Equation 7 of the paper.

The reasoner objective is:

`0.5 L_prefix + 0.3 L_stop + 0.3 L_counterfactual`.

There is no additional direct answer or aggregate-ranking loss. The
answer-support and cross-chain interaction heads remain part of the complete
chain score in Equation 10. Stop labels mark the earliest sufficient prefix on
each route. Every weak-positive training prefix uses all four counterfactual
types: time, relation, bridge/entity, and answer-tail.

Training artifacts use:

`outputs/runs/<stage>/main-3loss-multitq-10k/multitq-<hash>/`

Existing standalone retriever/reranker checkpoints in the legacy unlabeled
directory are reused; only the reasoner must be retrained.

From the project root:

```bash
bash multitq_src/scripts/run_main_experiment.sh
```

All long stages use `--resume auto`; rerun the same command after interruption.
