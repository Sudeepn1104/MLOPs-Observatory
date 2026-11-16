# The model: choices, evidence, and what they mean

The code is in [`model.py`](../model.py). The comparison behind every claim here is reproducible: run `python compare_models.py` and it logs each candidate to MLflow under the experiment `model-selection`.

## What the model is

- **Gradient boosting** (scikit-learn's `HistGradientBoostingClassifier`), with hyperparameters from a randomized, 5-fold cross-validated search.
- Wrapped in **`TunedThresholdClassifierCV`**, which picks the decision threshold by cross-validation to maximise F1.

So the model has two outputs, and they mean different things:

- `predict_proba` is a **calibrated probability**: of clients given 25%, about 25% default.
- `predict` is the **decision**: flag the client if the probability reaches the threshold (about 0.27 for v1).

## The comparison

Same training data, same 5 folds, same test split, same threshold tuning. Only the algorithm differs.

| Model | CV AUC (± std) | Test AUC | Calibration error (ECE) | Brier | F1 at tuned threshold | Size | 1-row latency |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.717 ± 0.013 | 0.719 | 0.241 | 0.209 | 0.503 | tiny | fast |
| RandomForest, balanced weights (previous model) | 0.770 ± 0.014 | 0.766 | 0.213 | 0.183 | 0.534 | 5.4 MB | ~47 ms |
| RandomForest + isotonic calibration | 0.771 ± 0.015 | 0.765 | 0.017 | 0.136 | 0.534 | larger | slower |
| Gradient boosting, scikit-learn defaults | 0.771 ± 0.014 | 0.769 | 0.010 | 0.136 | 0.522 | 0.06 MB | ~1 ms |
| **Gradient boosting, tuned (current)** | **0.773 ± 0.013** | 0.765 | 0.012 | 0.137 | 0.522 | 0.06 MB | ~1 ms |

## What the numbers say

**1. The algorithm barely matters for ranking.** All tree models are within 0.003 cross-validated AUC, and the noise between folds is ±0.013. The ceiling is the information in these 10 features, not the model. To do better you'd need better features, such as payment history ratios or the other 13 raw columns. A bigger model won't help.

**2. The previous model's probabilities were wrong.** Balanced class weights tell the model to treat the rare class (22% default) as if it were half the data. Ranking survives that, but every probability is inflated: the old model said "45%" when about 22% defaulted. Its calibration error was 0.21.

Probabilities matter as soon as anyone uses them as numbers: pricing, expected loss, comparing risk across groups, or a human reading "45%" on a screen. A mis-calibrated model is quietly misleading.

**3. Calibration can be fixed two ways, and they perform the same.** Calibrating the RandomForest afterwards (isotonic) gets it to 0.017. Gradient boosting is close to calibrated on its own (0.012), because it directly minimises log-loss and isn't reweighted. On accuracy, the two are equivalent.

**4. So the choice of gradient boosting is an engineering decision:**
- **No extra calibration layer**, so one fewer component to train, store and get wrong.
- **Explanations match what's served.** SHAP explains the model it's given. On a calibrated forest, SHAP would explain the uncalibrated forest underneath, not the probability the API returns.
- **About 90× smaller** (0.06 MB vs 5.4 MB) and much faster to score a single request, which matters for an API.

**5. Hyperparameter tuning bought nothing measurable** (+0.003 CV AUC, inside the noise; slightly lower on the test split). It's kept because it's cheap (about 15 s) and logged, so the evidence is on record, not because it helped. Reporting that honestly is part of the job.

## The decision threshold

A probability isn't a decision. Before, the threshold was 0.5 and the class weights silently moved it. Now it's explicit: `TunedThresholdClassifierCV` tries thresholds on cross-validation folds and keeps the one with the best F1 (about 0.27 for v1).

F1 treats a missed defaulter and a false alarm as equally bad. A lender wouldn't: a missed default loses money, and a false alarm loses a customer. With real costs you'd tune for expected profit instead, by changing the `scoring` argument. Each retrain re-tunes the threshold on its own data, and each registered version stores its threshold (shown in the registry table and returned by the API).

## Explanations (SHAP)

For one client, SHAP splits the model's output into one contribution per feature, in **log-odds**:

```
log-odds(default) = base value + contribution(credit_limit) + contribution(age) + ...
probability       = 1 / (1 + e^(-log-odds))
```

The contributions add up exactly to the prediction; a test checks this. Positive pushes towards default, negative away from it. Averaging their absolute size over many clients gives a global importance ranking. Recent repayment status (`repay_status_sep`) dominates, which is also why the simulated concept drift, which changes what repayment status means, breaks the model so badly.

Explanations are available:
- in the dashboard's **Explain** tab (global importance, plus a per-client chart)
- from the API with `POST /predict?explain=true`

**Limits of SHAP to say out loud:** it describes what the *model* does, not what *causes* default. Correlated features share credit in ways that can look arbitrary. And log-odds aren't intuitive, so the dashboard also shows the resulting probability.

## Fairness across age groups

`model.fairness()` splits the test set into four age groups and reports, per group: default rate, mean predicted risk, selection rate (share flagged), recall (share of real defaulters caught), false-positive rate and AUC. It then reports three summary checks:

| Check | What it asks | v1 | Previous model |
|---|---|---|---|
| Selection-rate ratio ≥ 0.8 (four-fifths rule) | Are groups flagged at similar rates? | **0.77, fails** | 0.83, passes |
| Recall gap ≤ 0.1 (equal opportunity) | Are real defaulters caught equally in every group? | 0.05, passes | 0.03, passes |
| Calibration gap ≤ 0.05 | Does "30% risk" mean 30% in every group? | 0.01, passes | **~0.22, fails** |

The real default rates differ by age: 0.19 for 30–39, 0.23–0.24 for the others. A model that is calibrated within each group flags each group at about its own rate, so the selection rates differ and the four-fifths check fails. The previous model "passed" that check only because it over-predicted risk for everyone by about 20 points.

This is a known result: when base rates differ, calibration within groups and equal selection rates can't both hold (Kleinberg, Mullainathan and Raghavan, 2016; Chouldechova, 2017). No model fixes it; you choose which property matters for the decision at hand. That's a policy and legal question, so the model card reports all three checks instead of quietly optimising one.

Also worth saying: the 50+ group has only 336 test clients, so its numbers carry real uncertainty. And age is a feature here; many jurisdictions restrict its use in credit decisions.

## Questions an interviewer might ask

- *Why gradient boosting?* Not for accuracy, which is equal across tree models. For calibrated probabilities without an extra layer, explanations that match the served output, and a much smaller, faster model.
- *Why not class weights for the imbalance?* They distort probabilities. Train on the real distribution and set the decision threshold explicitly instead.
- *How did you choose the threshold?* Cross-validated for F1. In practice I'd use the business's costs of each error type.
- *How do you know the probabilities are right?* Calibration error and Brier score on a held-out split, and a reliability plot in the model card.
- *Is the model fair?* It depends on the definition: it passes recall parity and per-group calibration, but fails the four-fifths rule because real default rates differ by age. I'd surface that trade-off to the people accountable for the decision.
- *What would improve it?* Better features, not a different algorithm. The comparison shows the algorithms are tied.
