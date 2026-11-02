# Limitations of the CargoMind models and of this evaluation

Each item states the limitation, the evidence for it, and what it means for how the results can be used. File paths are relative to `ML/Evaluation/`.

## A. Data

1. **The data is synthetic.**
   - Delay is generated as a sum of operational terms plus noise (`ML/Dataset/Raw/generate_dataset.py`). The models recover that structure, and that is all the results show.
   - No metric here is evidence of real-world performance.
2. **Destination state at departure is a generator artefact.**
   - The destination-state group is a top-2 permutation group for every model (`ML/ML/results/feature_importance/group_permutation_importance.csv`), even though the EDA found it weak on its own.
   - The generator uses destination conditions at the departure hour. Real delays would depend on conditions at *arrival*, which are unknown at prediction time.
   - Expect this signal to be weaker or absent in real data.
3. **The test period is short and narrow.**
   - It covers 11 departure days (2025-02-01 → 02-11) and 215 Critical shipments.
   - Per-port and per-carrier Critical counts are mostly in single or low double digits (`validation/slice_performance.csv`).
4. **There are no Critical air shipments in the test period** (0 of 128).
   - Air-freight Critical performance is untested.
   - The air macro-F1 of 0.33 is driven by two classes only.
5. **295 shipments with no observed outcome were dropped** by the ML stage.
   - If missing outcomes are not random (for example, long delays still in transit when the data was extracted), the class mix is biased toward shorter delays.
6. **Filled / reconstructed inputs.** Excluding rows with filled inputs changes nothing material (`validation/robustness_analysis.csv`), so this is a minor concern.

## B. Temporal design and labels

7. **Labels were used before they would be known.**
   - In every tuning fold, 42–62 % of the fold's training labels came from shipments that had not arrived by the fold's validation start (`validation/split_verification.json`).
   - CV scores, and the hyperparameter choices based on them, are therefore optimistic relative to real retraining.
   - The test-period sensitivity run (training only on labels known at test start) costs LR 0.022 macro-F1, and the tree models 0.12–0.18 Critical recall.
8. **The historical-delay features are censored.**
   - The "arrived before cutoff" history is biased toward fast shipments early in the dataset.
   - These features drift with PSI 2.5–4.0 between train and test (`metrics/feature_drift_train_vs_test.csv`).
   - Removing them improves LR (+0.011 macro-F1, interval excludes 0).
9. **Class prevalence is non-stationary.**
   - Critical share is 17.3 % in train, 23–26 % in the CV blocks and 9.0 % in test, with daily values from 3 % to 43 %.
   - Precision, F1 and calibration are therefore period-specific. LR Critical precision would be 0.73 at training prevalence but 0.40 at 5 % (`metrics/critical_precision_vs_prevalence.csv`).
   - Validation-tuned thresholds and recalibrators inherit the validation prevalence. This is why the cost-minimising thresholds and vector scaling transfer poorly to test.
10. **There is a single split.**
    - There is one test window and no rolling-origin backtest.
    - Stability across *test* periods is judged from day-level breakdowns only, and those are noisy.
11. **Most test outcomes are not yet known.** Only 20.6 % of test labels are known by the end of the test period. A live system could not have measured this performance until about mid-March.

## C. Models and probabilities

12. **Class weighting distorts the probabilities.**
    - All classifiers use balanced weights, so the raw probabilities over-state Critical on the test period (mean 0.13–0.16 vs 9 % observed).
    - A prior correction fixes most of this, but only for a known prevalence.
13. **Random Forest probabilities are poorly scaled.** It has the worst log loss (0.700) and ECE (0.134), and is strongly under-confident. It is unsuitable as a risk score without recalibration.
14. **The At-Risk class is inherently noisy.** Errors concentrate within a few hours of the 12 h and 48 h boundaries (`metrics/boundary_analysis.csv`). Hard class cut-offs on a continuous outcome put a ceiling on per-class accuracy.
15. **Detection is weaker in quiet periods.** LR Critical recall is 0.83 with an active origin disruption and 0.47 without one. Critical delays that build up from moderate conditions are the main failure mode.
16. **LR uses identity one-hots.** Removing port and carrier identity improves LR, which suggests mild overfitting to entities seen in training. New ports or carriers fall back to the baseline category (an all-zero one-hot); this is checked to run, but not checked for accuracy.

## D. Anomaly detection

17. **There is no ground truth for "unusual port conditions".**
    - The detectors are scored against shipment delay labels as a proxy.
    - Low recall against delays is expected and is not a failure of anomaly detection itself.
18. **Validation metrics for the detectors are partly in-sample.** The final detectors were fitted on all training On-Time rows, including the validation block (On-Time flag rate 5.1 % on validation vs 1.5 % on test).
19. **The detectors only see the origin port.** Delays caused at the destination or hub cannot be flagged.
20. **The flag threshold is a fixed percentile** of training On-Time scores. It will flag more or less often as port conditions drift. This was not tuned or monitored.

## E. Evaluation method

21. **There are few bootstrap clusters.**
    - Day-block intervals rest on 11 clusters, and percentile bootstraps with so few clusters tend to under-cover.
    - The port-day scheme (110 clusters) assumes ports are independent within a day, which is optimistic.
    - Treat every interval as approximate, and small differences (< 0.02 macro-F1) as ties.
22. **The cost matrix is assumed.**
    - The "minimum expected cost" threshold uses illustrative costs (missing a Critical shipment = 10× a false Critical alarm), not real ones.
    - Its validation-chosen thresholds would produce 200–630 Critical alerts per 1,000 shipments on test.
23. **There is no causal interpretation.** Feature importance and the ablation deltas describe what the models use on this data, not what causes delays.
24. **Some evaluations needed refits.**
    - Out-of-fold validation predictions, the ablation predictions, the final-model refit check and the anomaly stability/cost runs all use evaluation-only refits of the saved configurations.
    - Each refit reproduced the stored scores exactly, but these refits are not the saved models.
25. **Cost measurements are machine-specific.** Fit and scoring times were measured on one local machine (macOS arm64) at small scale. They do not predict cost at production volume.
