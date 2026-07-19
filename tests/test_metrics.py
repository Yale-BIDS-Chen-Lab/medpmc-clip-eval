import numpy as np
from medpmc_clip_eval.metrics import notebook_metrics


def test_binary_pooled_auc():
    y_true = np.array([0, 1])
    y_score = np.array([[0.9, 0.1], [0.2, 0.8]])
    acc, f1, auc = notebook_metrics(y_true, y_score, "binary")
    assert acc == 1.0
    assert f1 == 1.0
    assert auc == 1.0


def test_multilabel_mean_auc():
    y_true = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_score = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.8], [0.2, 0.1]])
    acc, f1, auc = notebook_metrics(y_true, y_score, "multilabel")
    assert acc == 1.0
    assert f1 == 1.0
    assert auc == 1.0


def test_specialty_aggregation():
    from medpmc_clip_eval.metrics import summarize_specialties

    rows = [
        {"dataset": "a", "specialty": "x", "task": "binary", "n": 10, "accuracy": 0.2, "f1": 0.3, "auc": 0.4},
        {"dataset": "b", "specialty": "x", "task": "binary", "n": 10, "accuracy": 0.4, "f1": 0.5, "auc": 0.6},
        {"dataset": "c", "specialty": "y", "task": "binary", "n": 10, "accuracy": 0.8, "f1": 0.7, "auc": 0.9},
        {"dataset": "OVERALL", "specialty": "", "task": "unweighted dataset mean", "n": 30, "accuracy": 0.4667, "f1": 0.5, "auc": 0.6333},
    ]

    out = summarize_specialties(rows, ci=False, confidence=0.95, n_bootstrap=100)
    by_name = {row["specialty"]: row for row in out}
    assert by_name["x"]["n_benchmarks"] == 2
    assert abs(by_name["x"]["accuracy"] - 0.3) < 1e-12
    assert abs(by_name["OVERALL_SPECIALTY_MACRO"]["accuracy"] - 0.55) < 1e-12


def test_weighted_auc_matches_repeated_samples():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from medpmc_clip_eval.metrics import WeightedAUC

    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.3, 0.8, 0.4])
    w = np.array([2, 3, 1, 4, 2])
    repeated_y = np.repeat(y, w)
    repeated_s = np.repeat(s, w)
    assert abs(WeightedAUC(y, s)(w) - roc_auc_score(repeated_y, repeated_s)) < 1e-12


def test_fast_npz_bootstrap_shapes():
    import numpy as np
    from medpmc_clip_eval.metrics import bootstrap_npz

    pred = {
        "format": "npz",
        "task": "binary",
        "dataset": "toy",
        "y_true": np.array([0, 0, 1, 1]),
        "y_score": np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]),
    }
    acc, f1, auc = bootstrap_npz(pred, n_bootstrap=5, seed=1)
    assert acc.shape == f1.shape == auc.shape == (5,)
    assert np.isfinite(auc).all()
