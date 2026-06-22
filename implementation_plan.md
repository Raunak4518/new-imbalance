# Add Direct wget/curl Downloads to All 40 Scripts

## Problem
Kaggle Notebooks block their own API from downloading datasets (403 Forbidden). All 3 fallback methods (CLI, kagglehub, opendatasets) fail. However, `pip install` and `wget` to **non-Kaggle URLs** work perfectly — proving internet is ON.

## Solution
For every script: download from the **original source** (official website / S3 / GitHub), NOT from Kaggle's API.

---

## Dataset Download Source Map

### Category A: Direct Public URL Available (wget works)

| # | Dataset | Direct URL | File(s) |
|---|---------|-----------|---------|
| 1 | ISIC_2019 | `isic-challenge-data.s3.amazonaws.com/2019/` | CSV + ZIP ✅ Done |
| 2 | ISIC_2020 | `isic-challenge-data.s3.amazonaws.com/2020/` | CSV + JPEG ZIP |
| 3 | Malaria_Cell | `data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip` | ImageFolder ZIP |
| 4 | Kvasir | `datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip` | ImageFolder ZIP |
| 5 | HyperKvasir | `datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip` | ImageFolder ZIP |
| 6 | NCT-CRC-HE-100K | `zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip` | ImageFolder ZIP |
| 7 | COVID-19_Radiography | GitHub release / Kaggle mirror | 4-class ImageFolder |
| 8 | Blood_Cell_Detection | GitHub BCCD dataset | 4-class ImageFolder |

### Category B: Kaggle-Only — Must Attach via "+ Add Data"
> [!IMPORTANT]
> These datasets exist ONLY on Kaggle. There is no alternative public URL. The script must detect if they're attached and give clear instructions if not.

| # | Dataset | Kaggle Handle | Type |
|---|---------|--------------|------|
| 9 | HAM10000 | `kmader/skin-cancer-mnist-ham10000` | Dataset |
| 10 | ChestX-ray14 | `nih-chest-xrays/data` | Dataset |
| 11 | TB_Chest_Xray | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | Dataset |
| 12 | BreakHis | `ambarish/breakhis` | Dataset |
| 13 | ODIR-5K | `andrewmvd/ocular-disease-recognition-odir5k` | Dataset |
| 14 | BraTS_2020 | `awsaf49/brats20-dataset-training-validation` | Dataset |
| 15 | OASIS | `jboysen/mri-and-alzheimers` | Dataset |
| 16 | CT_Medical_Images | `kmader/siim-medical-image-analysis-tutorial` | Dataset |
| 17 | BUSI | `aryashah2k/breast-ultrasound-images-dataset` | Dataset |
| 18 | PolypGen | `debeshjha/polypgen` | Dataset |
| 19 | Bone_Age_RSNA | `kmader/rsna-bone-age` | Dataset |
| 20 | RSNA_Pneumonia | `rsna-pneumonia-detection-challenge` | Competition |
| 21 | VinDr-CXR | `vinbigdata-chest-xray-abnormalities-detection` | Competition |
| 22 | PCam | `histopathologic-cancer-detection` | Competition |
| 23 | PANDA_Challenge | `prostate-cancer-grade-assessment` | Competition |
| 24 | APTOS_2019 | `aptos2019-blindness-detection` | Competition |
| 25 | EyePACS | `diabetic-retinopathy-detection` | Competition |
| 26 | RSNA_Intracranial | `rsna-intracranial-hemorrhage-detection` | Competition |
| 27 | Cervical_Cancer_Intel | `intel-mobileodt-cervical-cancer-screening` | Competition |

### Category C: Requires Registration / Credentials
> [!WARNING]
> These datasets require manual registration on external sites. Script will print clear instructions.

| # | Dataset | Official Source | Issue |
|---|---------|----------------|-------|
| 28 | CheXpert | Stanford ML Group | Registration required |
| 29 | MIMIC-CXR | PhysioNet | Credential required |
| 30 | BIMCV_COVID19 | BIMCV portal | Registration required |
| 31 | PAD_UFES_20 | Mendeley Data | Redirect-based download |
| 32 | DDI | DDI project page | Manual download |
| 33 | Camelyon16 | Grand Challenge | Registration required |
| 34 | Camelyon17 | Grand Challenge | Registration required |
| 35 | MESSIDOR-2 | ADCIS | Registration required |
| 36 | STARE | Clemson University | Direct but slow |
| 37 | IDRiD | IEEE DataPort | Login required |
| 38 | FastMRI_Knee | NYU FastMRI | Registration required |
| 39 | LIDC-IDRI | TCIA | NBIA required |
| 40 | MURA | Stanford ML Group | Registration required |

---

## Proposed Changes

### For each script, the download section will follow this exact pattern:

```python
# 1. Check /kaggle/input for attached data (always first)
# 2. If Category A: wget from original source URL
# 3. If Category B: print "Click + Add Data → search for <handle>"
# 4. If Category C: print "Register at <url> and upload manually"
```

### Execution Approach

Since the user explicitly requested working on individual scripts (not generator), I will:
1. Update `script_template.py` with the clean download pattern
2. Create a Python helper script that patches each existing script in `d:\new imbalance\scripts\` in-place
3. Each script gets its own hardcoded URLs / instructions

---

## Verification Plan

### Automated
- `python -m py_compile` on every generated script

### Manual
- User uploads `run_isic_2019.py` to Kaggle → should auto-download via wget from S3
- User uploads a Category B script → should print clear "Add Data" instructions
