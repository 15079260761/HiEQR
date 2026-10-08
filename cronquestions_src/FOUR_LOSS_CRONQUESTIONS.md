# CronQuestions paper-aligned four-loss main experiment

The reasoner objective is:

`1.0 L_answer + 0.5 L_prefix + 0.3 L_stop + 0.3 L_counterfactual`.

The legacy aggregate ranking loss is disabled. Answer supervision jointly
trains the answer scorer and answer-centred chain interaction head. Stop labels
mark the earliest sufficient prefix on each route. Every weak-positive training
prefix uses all four distinct counterfactual types: time, relation, bridge/entity,
and answer-tail. Evaluation exports evidence-gain trajectories and explicit
cooperation/competition records.

Training artifacts use:

`cronquestions_outputs/runs/<stage>/main-4loss-cronquestions-10k/cronquestions-<hash>/`

Existing standalone retriever/reranker checkpoints and validated preprocessing
caches are reused. Only the reasoner must be retrained on the deterministic
10,000-question training sample, followed by the full 30,000-question test.

From the project root:

```bash
bash cronquestions_src/scripts/run_main_experiment.sh
```

All long stages use `--resume auto`; rerun the same command after interruption.
