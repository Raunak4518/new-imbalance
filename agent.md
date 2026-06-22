

Phase 4: Evaluation Protocol

For every experiment add 

Classification Metrics

* Accuracy
* Balanced Accuracy
* Precision Macro
* Precision Weighted
* Recall Macro
* Recall Weighted
* F1 Macro
* F1 Weighted
* MCC
* Cohen Kappa

Long-tail Metrics

* Head Accuracy
* Medium Accuracy
* Tail Accuracy
* Many-Shot Accuracy
* Medium-Shot Accuracy
* Few-Shot Accuracy

Ranking Metrics

* Top-1
* Top-3
* Top-5

Probabilistic Metrics

* Log Loss
* Brier Score
* ECE
* Calibration Error

ROC Metrics

* ROC-AUC Macro
* ROC-AUC Weighted
* ROC-AUC Per Class

PR Metrics

* Average Precision
* PR-AUC

Resource Metrics

* Training Time
* Inference Time
* FLOPs
* Parameter Count
* GPU Memory

---

Phase 5: Publication Quality Figures

Automatically generate and save:

Training Curves

* Loss Curve
* Accuracy Curve
* F1 Curve

Class Analysis

* Class Distribution
* Long Tail Distribution
* Per Class Accuracy

Confusion Analysis

* Confusion Matrix
* Normalized Confusion Matrix

Embedding Analysis

* t-SNE
* UMAP

Calibration

* Reliability Diagram
* Confidence Histogram

ROC

* ROC Curves
* PR Curves

Save:

```text
results/
    figures/
    metrics/
    checkpoints/
    predictions/
```

Export figures as:

* PNG
* PDF
* SVG

Publication quality:

* 300 DPI minimum
* vector format where possible





Phase 7: Statistical Analysis

Automatically perform:

* Mean ± Std
* Confidence Intervals
* Wilcoxon Signed Rank Test
* Paired t-test
* Effect Size

Generate:

```text
statistical_report.pdf
```

for publication.

---

Phase 8: Research Integrity

Requirements:

* Fixed random seed
* Deterministic training
* Version logging
* Hardware logging
* Reproducibility report
* Save exact configuration used for every run

Never alter architecture during experiments.

Goal:
Produce publication-quality benchmarking of a fixed long-tail medical image classification architecture across approximately 40 medical datasets with fully automated training, evaluation, visualization, statistical analysis, and reproducibility reporting.

