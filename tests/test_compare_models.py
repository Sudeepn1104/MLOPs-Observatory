from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import compare_models
import registry


def test_compare_runs_every_candidate_and_logs_to_mlflow(trained, monkeypatch):
    # the machinery is under test, not the real (slow) candidates
    fast = {"small tree": DecisionTreeClassifier(max_depth=3, random_state=0),
            "logistic": LogisticRegression(max_iter=500)}
    monkeypatch.setattr(compare_models, "candidates", lambda: fast)
    results = compare_models.main(sample=1500)
    assert list(results["model"]) == list(fast)
    assert {"cv_auc", "ece", "f1_at_tuned_threshold", "tuned_threshold"} <= set(results.columns)
    assert results["cv_auc"].between(0.55, 1).all()
    c = registry.client()
    exp = c.get_experiment_by_name(compare_models.EXPERIMENT)
    assert len(c.search_runs([exp.experiment_id])) == 2


def test_real_candidates_cover_the_documented_comparison():
    names = list(compare_models.candidates())
    assert any("random forest" in n for n in names)
    assert any("calibration" in n for n in names)
    assert any("tuned, current model" in n for n in names)
