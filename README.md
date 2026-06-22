# STTP-Net Long-Tail Imbalance Benchmark Suite

This repository contains the automated pipeline for validating the `STTP-Net` (Hybrid of Hybrids) architecture across a massive scale of 40 medical imaging datasets.

## Project Structure

- `datasets.csv`: The centralized registry of all 40 medical datasets including download methods, metrics, and Kaggle URLs.
- `script_template.py`: The master template that contains the **EXACT unmodified** `STTP-Net` architecture, loss functions, and evaluation protocol.
- `generator.py`: The automation script that scaffolds out individual scripts for every dataset.
- `scripts/`: The output directory containing 40 standalone monolithic Python scripts (e.g., `run_ham10000.py`, `run_isic_2019.py`).

## How to Run Experiments

Every script in the `scripts/` directory is completely autonomous and designed specifically for executing on **Kaggle Notebooks with Dual T4 GPUs** or local machines.

### 1. Kaggle Notebook Execution (Recommended)
When running inside a Kaggle environment:
1. Create a Kaggle Notebook and turn on **GPU T4 x2**.
2. **Attach the Dataset** to your notebook via the "Add Data" menu on the right.
3. Upload the generated script (e.g. `run_ham10000.py`) into your notebook workspace.
4. Run:
```bash
!python run_ham10000.py
```
**Features:**
- The script automatically detects the attached dataset mount at `/kaggle/input/...` so it won't waste time downloading it again.
- It automatically wraps the `STTPNet` model in `nn.DataParallel` and scales the batch size natively to utilize **both T4 GPUs** simultaneously, routing custom functions effectively!

### 2. Automated Downloads (Local Machines)
To run an experiment locally on a Kaggle dataset (e.g., HAM10000):
```bash
cd scripts
python run_ham10000.py
```
The script will automatically trigger the Kaggle CLI to download the dataset into a local `results_HAM10000/data` folder, unzip it, begin training the exact `STTP-Net` architecture, and save results.

### 3. Manual Credentials Required
Certain medical datasets (such as MIMIC-CXR from PhysioNet, or CheXpert) require strict credential access. If you run one of these scripts:
```bash
python run_mimic_cxr.py
```
The script will detect that manual download is required, print instructions with the target path (e.g., `results_MIMIC-CXR/data/images`), and safely exit. 
**Action Required:**
1. Download the images manually using your institutional credentials.
2. Place the images in the requested directory.
3. Organize the dataset into `Class_Name/image.jpg` format (PyTorch `ImageFolder` standard).
4. Run `python run_mimic_cxr.py` again. The script will detect the data and proceed to train.

## Architectural Integrity Guarantee
The core components defined in `script_template.py` (and subsequently injected into every generated script) have been extracted exactly from `STTP_Net_exp12_HybridOfHybrids.ipynb`:
- `Encoder`, `CVAE`, `Head`, `STTPNet` classes.
- `EBSLoss`, `SupConLoss` loss functions.
- `pixel_cutmix` and `latent_mixup` logic.

### Standardized Inputs
Because the STTP-Net encoder expects a strict spatial dimension, **ALL** datasets—regardless of their native dimensions (e.g., 512x512 X-rays)—are resized to `224x224` during the PyTorch transform step to maintain perfect architectural alignment.

## Outputs
For every run, the script will create a `results_{DATASET_NAME}` folder containing:
- `data/`: The raw images
- `experiments/`:
  - `best_model.pt`: Checkpoint of the highest Balanced Accuracy model.
  - `training_curves.png / .svg`: High-resolution vector and rasterized loss/accuracy curves.
  - `confusion_matrix.png / .pdf`: Publication quality normalized confusion matrix.
  - `statistical_report.txt`: A detailed report of Precision, Recall, F1, MCC, Cohen's Kappa, Brier Score, and per-class metrics.

## Statistical Analysis
By running all 40 scripts, you will generate 40 statistical reports. These metrics can then be extracted to compute the overall Wilcoxon Signed Rank and paired t-tests for publication.
