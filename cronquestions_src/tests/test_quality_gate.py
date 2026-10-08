from decgc.evaluation.quality_gate import evaluate_metric_gate


def test_quality_gate_uses_real_metrics_and_both_thresholds():
    passed = evaluate_metric_gate(
        {"hits1": 0.82, "hits10": 0.98},
        {"hits1": 0.82, "hits10": 0.98},
    )
    failed = evaluate_metric_gate(
        {"hits1": 0.81, "hits10": 0.99},
        {"hits1": 0.82, "hits10": 0.98},
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["hits1"]["actual"] == 0.81
    assert failed["source"] == "persisted_demo_metrics"
