
You are a research engineer working on validating a fixed long-tail class imbalance architecture for publication.

Objective:
Evaluate the exact same architecture across a large collection of medical imaging datasets without modifying the architecture itself.
D:\new imbalance\STTP_Net_exp12_HybridOfHybrids.ipynb this is the full architecture 

Critical Constraints:

1. The architecture, training procedure, loss functions, augmentation strategy, optimizer, scheduler, and all model components must remain EXACTLY unchanged.
2. Only dataset-specific adaptations are allowed:

   * Number of classes
   * Input image channels if necessary
   * Dataset path
   * Label mappings
3. Never introduce new techniques, layers, losses, samplers, balancing methods, or preprocessing steps unless explicitly requested.
4. Maintain scientific reproducibility.

Tasks:

Phase 1: Dataset Discovery

Research and create a database of at least 40 medical datasets relevant to image classification and long-tail imbalance evaluation.

For each dataset collect:

* Dataset Name
* Domain
* Number of Classes
* Number of Samples
* Modality

  * Dermoscopy
  * X-ray
  * Histopathology
  * MRI
  * CT
  * Ultrasound
  * Retinal Imaging
  * Endoscopy
* Dataset URL
* Download Method
* Kaggle Availability
* License
* Class Distribution
* Imbalance Ratio
* Paper Citation

Generate:

datasets.csv

containing all metadata.

Rank datasets by suitability for long-tail evaluation.

---

Phase 2: Experiment Framework Generation

Generate a single platform-independent Python framework.

Requirements:

1. Runs on:

   * Kaggle
   * Linux
   * Windows
   * WSL

2. User should only modify:

DATASET_PATH

at the top of the script.

3. All hyperparameters must be centralized in a configuration block:

```python
CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
    ...
}
```

4. Auto install dependencies:


a single script 

---

Phase 3: Dataset Download Automation

For every supported dataset create CLI download commands.

Examples:

Kaggle:

```bash
kaggle datasets download ...
```

Google Drive:

```bash
gdown ...
```

HuggingFace:

```bash
huggingface-cli download ...
```

Direct URL:

```bash
wget ...
```

Store all download commands inside:

download_dataset.py

User should only specify:

```python
DATASET_NAME = "HAM10000"
```

and download should happen automatically.

---

Phase 4: Evaluation Protocol

For every experiment compute:

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

