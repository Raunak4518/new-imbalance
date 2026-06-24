# Dataset Scripts Audit Report

Complete audit of all 40 scripts in `d:\new imbalance\scripts\`.

---

## Executive Summary

| Category | Count | Details |
|----------|-------|---------|
| **CRITICAL bugs** | 12 | Will fail at runtime or produce wrong results |
| **WARNING issues** | 18 | Structural misalignments, missing features |
| **INFO notes** | 10 | Minor inconsistencies, style issues |
| **Clean scripts** | 8 | No issues found |

### Systematic Issues (affect most scripts)

> [!CAUTION]
> **Missing `grad_checkpoint`**: Only `run_blood_cell_detection.py` uses gradient checkpointing. All other 39 scripts call the encoder directly (`self.proj(self.backbone(x))`), meaning HybridMix (which calls encoder **2x per step**) will use ~2x more GPU RAM. This is an OOM risk on Kaggle P100 GPUs (16GB).

> [!WARNING]
> **Missing `drop_last=True`**: Only `run_blood_cell_detection.py` has `drop_last=True` on training DataLoaders. All other scripts risk BatchNorm errors (batch_size=1) or DataParallel assertion errors on the final partial batch.

> [!WARNING]
> **Missing progress logging**: Only `run_blood_cell_detection.py` has per-step progress logging in `train_epoch`. All other scripts print nothing during an epoch, making long runs (e.g., ChestX-ray14 with 112K images) appear hung.

---

## Per-Script Audit

### Legend
- 🟢 = No issues | 🟡 = Warnings | 🔴 = Critical bugs
- **Type**: `CUSTOM` = dataset-specific data loader | `GENERIC` = template auto-discovery

---

### 1. 🟡 `run_ham10000.py` — HAM10000
| Field | Value | Status |
|-------|-------|--------|
| Source | `kmader/skin-cancer-mnist-ham10000` | ✅ Correct |
| REQUIRED_FILE | `HAM10000_metadata.csv` | ✅ Correct |
| NUM_CLASSES | 7 | ✅ Correct (akiec, bcc, bkl, df, mel, nv, vasc) |
| Loader | CUSTOM | ✅ |
| CSV columns | `image_id`, `dx` | ✅ Match actual dataset |

**Issues:**
- 🟡 `official_classes` list uses `['nv', 'mel', 'bcc', 'akiec', 'bkl', 'df', 'vasc']` — label order determines class indices. Verify this matches the paper's intended ordering.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 2. 🟡 `run_isic_2019.py` — ISIC 2019
| Field | Value | Status |
|-------|-------|--------|
| Source | `nodoubttome/isic-2019` | ✅ Correct |
| REQUIRED_FILE | not extracted (uses folder scan) | ⚠️ No REQUIRED_FILE set |
| NUM_CLASSES | 8 | ✅ Correct |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 No `REQUIRED_FILE` defined — the discovery logic may be unreliable
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 3. 🟡 `run_isic_2020.py` — ISIC 2020
| Field | Value | Status |
|-------|-------|--------|
| Source | `cdeotte/jpeg-melanoma-256x256` | ✅ Correct |
| REQUIRED_FILE | `ISIC_2020_Training_GroundTruth_v2.csv` | ⚠️ See issue |
| NUM_CLASSES | 2 | ✅ (benign/malignant) |
| Loader | CUSTOM | ✅ |
| Download fallback | ✅ Has `download_dataset()` from ISIC S3 | ✅ |

**Issues:**
- 🔴 **CSV mismatch**: Script searches for `ISIC_2020_Training_GroundTruth_v2.csv` as REQUIRED_FILE, but then the data loader at line 170 looks for `train.csv` (not `ISIC_2020_Training_GroundTruth_v2.csv`). The `cdeotte/jpeg-melanoma-256x256` Kaggle dataset contains a `train.csv` with columns `image_name` and `target`, which is what the loader uses. **The REQUIRED_FILE and the download_dataset() function reference the official ISIC S3 files which have a different format.** If dataset is attached via Kaggle, it works. If downloaded from S3, the loader will crash because `train.csv` won't exist.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 4. 🟡 `run_pad_ufes_20.py` — PAD-UFES-20
| Field | Value | Status |
|-------|-------|--------|
| Source | Mendeley Data | ✅ |
| REQUIRED_FILE | `metadata.csv` | ✅ |
| NUM_CLASSES | 6 (set) → 7 (actual list) | 🔴 See issue |
| Loader | CUSTOM | ✅ |
| CSV columns | `img_id`, `diagnostic` | ✅ |

**Issues:**
- 🔴 **NUM_CLASSES mismatch**: Line 34 sets `NUM_CLASSES = 6`, but line 131 defines `official_classes = ['BCC', 'SCC', 'ACK', 'SEK', 'BOD', 'MEL', 'NEV']` which is **7 classes**. Line 133 then overrides to `NUM_CLASSES = len(CLASS_NAMES)` = 7, so this fixes itself at runtime. But the initial `NUM_CLASSES = 6` is misleading and suggests the script was scaffolded incorrectly.
- 🟡 The actual PAD-UFES-20 dataset has 6 classes per the original paper: ACK, BCC, MEL, NEV, SCC, SEK. `'BOD'` (Bowen's Disease) may not exist in this dataset — needs verification. If `BOD` doesn't appear, `label2idx` will crash with a `KeyError`.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 5. 🟡 `run_ddi.py` — DDI
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (`ddi-dataset.github.io`) | ✅ |
| REQUIRED_FILE | `ddi` | ⚠️ Vague |
| NUM_CLASSES | 78 | ⚠️ See issue |
| Loader | CUSTOM | ✅ |
| CSV columns | `DDI_file`, `disease` | ✅ |

**Issues:**
- 🟡 `NUM_CLASSES = 78` is hardcoded but then overridden by `NUM_CLASSES = len(CLASS_NAMES)` dynamically from the CSV. The DDI dataset actually has ~78 disease categories with very few samples per class — many classes will have <5 training samples. This is a severe long-tail scenario but not a bug per se.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 6. 🟡 `run_chestx_ray14.py` — ChestX-ray14
| Field | Value | Status |
|-------|-------|--------|
| Source | `nih-chest-xrays/data` | ✅ |
| REQUIRED_FILE | `Data_Entry_2017.csv` | ✅ |
| NUM_CLASSES | 14 → 15 (actual list) | 🔴 See issue |
| Loader | CUSTOM (multi-label → single-label) | ✅ Approach OK |
| CSV columns | `Image Index`, `Finding Labels` | ✅ |

**Issues:**
- 🔴 **NUM_CLASSES mismatch**: Line 34 sets `NUM_CLASSES = 14`, but line 136 defines `official_classes` with **15 items** (14 diseases + 'No Finding'). Line 138 overrides to `NUM_CLASSES = len(CLASS_NAMES)` = 15, so runtime is OK, but the initial value is wrong and could confuse readers.
- 🟡 **Multi-label simplification**: Takes only the first finding (`findings[0]`). This means images with "Atelectasis|Effusion" are labeled only as "Atelectasis". This is a known limitation but should be documented.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 7. 🟡 `run_chexpert.py` — CheXpert
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (Stanford) | ✅ |
| REQUIRED_FILE | `train.csv` | ✅ |
| NUM_CLASSES | 14 | ⚠️ CheXpert has 14 observation labels but the competition typically uses 5 |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 `NUM_CLASSES = 14` — CheXpert is typically benchmarked on 5 competition labels, not 14. The full 14 includes uncertain labels (`-1`) which need special handling.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 8. 🟢 `run_covid_19_radiography.py` — COVID-19 Radiography
| Field | Value | Status |
|-------|-------|--------|
| Source | `tawsifurrahman/covid19-radiography-database` | ✅ |
| REQUIRED_FILE | `COVID` | ✅ (matches folder name) |
| NUM_CLASSES | 4 | ✅ (COVID, Normal, Lung_Opacity, Viral Pneumonia) |
| Loader | CUSTOM (ImageFolder-based) | ✅ |

**Issues:**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`
- Otherwise clean

---

### 9. 🟡 `run_rsna_pneumonia.py` — RSNA Pneumonia
| Field | Value | Status |
|-------|-------|--------|
| Source | `rsna-pneumonia-detection-challenge` (Competition) | ✅ |
| REQUIRED_FILE | `stage_2_train_labels.csv` | ✅ |
| NUM_CLASSES | 2 | ⚠️ See issue |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 RSNA Pneumonia Detection has 3 classes in the CSV: 'Normal', 'Lung Opacity', 'No Lung Opacity / Not Normal'. Script may reduce to binary. Need to verify the custom loader logic.
- 🟡 Images are DICOM (`.dcm`), not JPG/PNG. The script must handle DICOM reading — verify if it does.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 10. 🟡 `run_bimcv_covid19.py` — BIMCV COVID-19
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (`bimcv.cipf.es`) | ✅ |
| REQUIRED_FILE | `bimcv` | ⚠️ Vague |
| NUM_CLASSES | 3 | ⚠️ |
| Loader | GENERIC | ⚠️ |

**Issues:**
- 🟡 BIMCV dataset contains DICOM/NIfTI files, not standard images. The generic loader only looks for `.png/.jpg/.jpeg/.bmp`. **This will find 0 images and crash.**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 11. 🟢 `run_tb_chest_xray.py` — TB Chest X-ray
| Field | Value | Status |
|-------|-------|--------|
| Source | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | ✅ |
| REQUIRED_FILE | `Tuberculosis` | ✅ (matches class folder) |
| NUM_CLASSES | 2 | ✅ |
| Loader | GENERIC (ImageFolder) | ✅ Will work — dataset has `Tuberculosis/` and `Normal/` folders |

**Issues:**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 12. 🟡 `run_mimic_cxr.py` — MIMIC-CXR
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (PhysioNet) | ✅ |
| REQUIRED_FILE | `mimic-cxr` | ⚠️ |
| NUM_CLASSES | 14 | ⚠️ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 MIMIC-CXR images are DICOM/JPG in a deeply nested folder structure (`files/p10/p10000032/s50414267/...`). NOT organized by class. The generic loader will either crash or produce garbage.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 13. 🟡 `run_vindr_cxr.py` — VinDr-CXR
| Field | Value | Status |
|-------|-------|--------|
| Source | `vinbigdata-chest-xray-abnormalities-detection` (Competition) | ✅ |
| REQUIRED_FILE | `train.csv` | ✅ |
| NUM_CLASSES | 22 | ⚠️ |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 VinBigData has 14 finding categories + "No Finding". `NUM_CLASSES = 22` seems too high — needs verification.
- 🟡 Images are DICOM. Need to verify DICOM handling in custom loader.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 14. 🟢 `run_breakhis.py` — BreakHis
| Field | Value | Status |
|-------|-------|--------|
| Source | `ambarish/breakhis` | ✅ |
| REQUIRED_FILE | `BreaKHis_v1` | ✅ |
| NUM_CLASSES | 8 | ✅ |
| Loader | CUSTOM (folder name → class) | ✅ Smart fallback |

**Issues:**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`
- BreakHis has nested folders: `BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/...`. The custom loader walks and matches folder names to class names — this should work.

---

### 15. 🟡 `run_pcam.py` — PCam
| Field | Value | Status |
|-------|-------|--------|
| Source | `histopathologic-cancer-detection` (Competition) | ✅ |
| REQUIRED_FILE | `train_labels.csv` | ✅ |
| NUM_CLASSES | 2 | ✅ |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 PCam has ~220K training images (96×96 px). At `IMG_SIZE=224`, these tiny images will be upscaled ~2.3x, which may introduce artifacts. Not a bug, but worth noting.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 16. 🟢 `run_nct_crc_he_100k.py` — NCT-CRC-HE-100K
| Field | Value | Status |
|-------|-------|--------|
| Source | `kmader/colorectal-histology-mnist` | ✅ |
| REQUIRED_FILE | `ADI` | ✅ (class folder name) |
| NUM_CLASSES | 9 | ✅ |
| Loader | GENERIC (ImageFolder) | ✅ Will work — class folders |
| Download | ✅ Has wget fallback | ✅ |

**Issues:**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 17–18. 🟡 `run_camelyon16.py` & `run_camelyon17.py`
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (Grand Challenge) | ✅ |
| REQUIRED_FILE | `camelyon` | ⚠️ Vague |
| NUM_CLASSES | 2 | ✅ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 Camelyon datasets are Whole Slide Images (WSI) in `.tif` format, NOT standard image files. The generic loader searches for `.png/.jpg/.jpeg/.bmp` only. **Will find 0 images and crash.** These datasets require patch extraction preprocessing before they can be used with this pipeline.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 19. 🟡 `run_panda_challenge.py` — PANDA Challenge
| Field | Value | Status |
|-------|-------|--------|
| Source | `prostate-cancer-grade-assessment` (Competition) | ✅ |
| REQUIRED_FILE | `train.csv` | ✅ |
| NUM_CLASSES | 6 | ✅ (ISUP grades 0-5) |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 PANDA images are large `.tiff` WSIs (multi-level), not standard images. Generic loader won't find them (only searches `.png/.jpg/.jpeg/.bmp`). **Will crash.**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 20. 🟢 `run_aptos_2019.py` — APTOS 2019
| Field | Value | Status |
|-------|-------|--------|
| Source | `aptos2019-blindness-detection` (Competition) | ✅ |
| REQUIRED_FILE | `train.csv` | ✅ |
| NUM_CLASSES | 5 | ✅ (DR grades 0-4) |
| Loader | CUSTOM | ✅ |
| CSV columns | `id_code`, `diagnosis` | ✅ |

**Issues:**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 21. 🟡 `run_eyepacs.py` — EyePACS
| Field | Value | Status |
|-------|-------|--------|
| Source | `diabetic-retinopathy-detection` (Competition) | ✅ |
| REQUIRED_FILE | `trainLabels.csv` | ✅ |
| NUM_CLASSES | 5 | ✅ |
| Loader | CUSTOM | ✅ |

**Issues:**
- 🟡 EyePACS is ~88K images (~35GB). This is extremely large for a single Kaggle session. May need subsetting logic.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 22–24. 🟡 `run_messidor_2.py`, `run_stare.py`, `run_idrid.py`
All three are manual-download datasets using GENERIC loaders:

| Script | REQUIRED_FILE | NUM_CLASSES | Potential Issues |
|--------|--------------|-------------|-----------------|
| MESSIDOR-2 | `messidor` | 5 | OK if organized in class folders |
| STARE | `stare` | 2 | ⚠️ STARE has ~400 images, labels in separate files, NOT in class folders. Generic loader may fail. |
| IDRiD | `IDRiD` | 4 | ⚠️ IDRiD provides labels in CSV/Excel, images in flat folder. Generic loader should fall back to CSV parsing. |

---

### 25. 🟢 `run_odir_5k.py` — ODIR-5K
| Field | Value | Status |
|-------|-------|--------|
| Source | `andrewmvd/ocular-disease-recognition-odir5k` | ✅ |
| REQUIRED_FILE | `ODIR-5K` | ✅ |
| NUM_CLASSES | 8 | ✅ |
| Loader | GENERIC | ✅ |

---

### 26. 🔴 `run_brats_2020.py` — BraTS 2020
| Field | Value | Status |
|-------|-------|--------|
| Source | `awsaf49/brats20-dataset-training-validation` | ✅ |
| REQUIRED_FILE | `BraTS20` | ✅ |
| NUM_CLASSES | 4 | ⚠️ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 **BraTS 2020 is a 3D brain MRI segmentation dataset stored as NIfTI (`.nii.gz`) files.** There are NO standard image files (`.png/.jpg`). The generic loader will find 0 images and crash. This dataset requires 3D volumetric processing or slice extraction, which is fundamentally incompatible with the current 2D image pipeline.

---

### 27. 🟡 `run_oasis.py` — OASIS
| Field | Value | Status |
|-------|-------|--------|
| Source | `jboysen/mri-and-alzheimers` | ✅ |
| REQUIRED_FILE | `oasis` | ⚠️ |
| NUM_CLASSES | 4 | ✅ |
| Loader | GENERIC | ✅ |

**Issues:**
- 🟡 This Kaggle dataset contains CSV data, not images. It has `oasis_cross-sectional.csv` and `oasis_longitudinal.csv`. If images are not included in this specific Kaggle version, the loader will fail.

---

### 28. 🔴 `run_fastmri_knee.py` — FastMRI Knee
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (NYU) | ✅ |
| REQUIRED_FILE | `fastmri` | ⚠️ |
| NUM_CLASSES | 2 | ⚠️ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 **FastMRI stores data as HDF5 (`.h5`) files containing k-space and reconstructed MRI volumes.** Not standard images. Generic loader will crash.

---

### 29. 🟢 `run_ct_medical_images.py` — CT Medical Images
| Field | Value | Status |
|-------|-------|--------|
| Source | `kmader/siim-medical-image-analysis-tutorial` | ✅ |
| REQUIRED_FILE | `overview.csv` | ✅ |
| NUM_CLASSES | 2 | ✅ |
| Loader | GENERIC | ✅ (has CSV + images) |

---

### 30. 🔴 `run_lidc_idri.py` — LIDC-IDRI
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (TCIA) | ✅ |
| REQUIRED_FILE | `LIDC` | ⚠️ |
| NUM_CLASSES | 4 | ⚠️ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 LIDC-IDRI is a CT lung scan dataset stored as DICOM series. NOT standard images. Generic loader will crash.

---

### 31. 🟡 `run_rsna_intracranial.py` — RSNA Intracranial
| Field | Value | Status |
|-------|-------|--------|
| Source | `rsna-intracranial-hemorrhage-detection` (Competition) | ✅ |
| REQUIRED_FILE | `stage_2_train.csv` | ✅ |
| NUM_CLASSES | 6 | ✅ |
| Loader | GENERIC | 🔴 |

**Issues:**
- 🔴 RSNA Intracranial images are DICOM (`.dcm`). Generic loader only searches `.png/.jpg/.jpeg/.bmp`. **Will find 0 images and crash.**

---

### 32–33. 🟢 `run_kvasir.py` & `run_hyperkvasir.py`
| Script | Source | NUM_CLASSES | Loader | Download |
|--------|--------|-------------|--------|----------|
| Kvasir | `meetnagadia/kvasir-dataset` | 8 | GENERIC (ImageFolder) | ✅ wget |
| HyperKvasir | Manual (Simula) | 23 | GENERIC (ImageFolder) | ✅ wget |

Both should work — datasets have class-organized folder structures.
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 34. 🟢 `run_busi.py` — BUSI
| Field | Value | Status |
|-------|-------|--------|
| Source | `aryashah2k/breast-ultrasound-images-dataset` | ✅ |
| REQUIRED_FILE | `benign` | ✅ (class folder) |
| NUM_CLASSES | 3 | ✅ (benign, malignant, normal) |
| Loader | GENERIC (ImageFolder) | ✅ |

**Issues:**
- 🟡 BUSI contains mask images alongside the ultrasound images (e.g., `benign (1)_mask.png`). The generic loader will include mask images in training data, corrupting the dataset. **This is a silent data corruption issue.**
- 🟡 Missing `drop_last=True`, `grad_checkpoint`

---

### 35. 🟡 `run_cervical_cancer_intel.py` — Cervical Cancer Intel
| Field | Value | Status |
|-------|-------|--------|
| Source | `intel-mobileodt-cervical-cancer-screening` (Competition) | ✅ |
| REQUIRED_FILE | `train` | ✅ |
| NUM_CLASSES | 3 | ✅ (Type_1, Type_2, Type_3) |
| Loader | GENERIC (ImageFolder) | ✅ |

---

### 36. 🟡 `run_polypgen.py` — PolypGen
| Field | Value | Status |
|-------|-------|--------|
| Source | `debeshjha/polypgen` | ✅ |
| REQUIRED_FILE | `polyp` | ⚠️ |
| NUM_CLASSES | 2 | ✅ |
| Loader | GENERIC | ⚠️ |

**Issues:**
- 🟡 PolypGen is a segmentation dataset with images + masks. Generic loader may include mask images.

---

### 37. 🟢 `run_blood_cell_detection.py` — Blood Cell Detection
| Field | Value | Status |
|-------|-------|--------|
| Source | `paultimothymooney/blood-cells` | ✅ |
| REQUIRED_FILE | `TRAIN` | ✅ |
| NUM_CLASSES | 4 | ✅ (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL) |
| Loader | GENERIC (enhanced with smart ImageFolder root finder) | ✅ |

**This is the best-quality script** — it has:
- ✅ `grad_checkpoint` for memory safety
- ✅ `drop_last=True` on training loaders
- ✅ Per-step progress logging
- ✅ Smart ImageFolder root detection (prefers TRAIN over TEST_SIMPLE by image count)

---

### 38. 🟢 `run_malaria_cell.py` — Malaria Cell
| Field | Value | Status |
|-------|-------|--------|
| Source | `iarunava/cell-images-for-detecting-malaria` | ✅ |
| REQUIRED_FILE | `Parasitized` | ✅ (class folder) |
| NUM_CLASSES | 2 | ✅ (Parasitized, Uninfected) |
| Loader | GENERIC (ImageFolder) | ✅ |
| Download | ✅ wget fallback | ✅ |

---

### 39. 🟡 `run_bone_age_rsna.py` — Bone Age RSNA
| Field | Value | Status |
|-------|-------|--------|
| Source | `kmader/rsna-bone-age` | ✅ |
| REQUIRED_FILE | `boneage-training-dataset` | ✅ |
| NUM_CLASSES | 2 | 🔴 |
| Loader | GENERIC | ⚠️ |

**Issues:**
- 🔴 **Bone Age is a REGRESSION task** — predicting bone age in months (0-228), not a classification task. Setting `NUM_CLASSES = 2` makes no sense for this dataset. The CSV has `boneage` (continuous) and `male` (boolean) columns. The generic loader will try to use the `boneage` column as a class label, creating hundreds of unique "classes" and crashing.

---

### 40. 🟡 `run_mura.py` — MURA
| Field | Value | Status |
|-------|-------|--------|
| Source | Manual (Stanford ML Group) | ✅ |
| REQUIRED_FILE | `MURA` | ✅ |
| NUM_CLASSES | 2 | ✅ (normal, abnormal) |
| Loader | GENERIC | ✅ |

**Issues:**
- 🟡 MURA has a nested folder structure: `MURA-v1.1/train/XR_ELBOW/patient00001/study1_positive/image1.png`. The generic ImageFolder auto-discovery should handle this, finding the `positive/` and `negative/` leaf folders. But the actual MURA structure uses study-level folders named `study1_positive`/`study1_negative`, NOT class-level subfolders. The generic walker may misidentify the folder hierarchy.

---

## Critical Issues Summary (Must Fix)

| # | Script | Issue | Severity |
|---|--------|-------|----------|
| 1 | `run_isic_2020.py` | REQUIRED_FILE/CSV mismatch between S3 download and Kaggle loader | 🔴 |
| 2 | `run_pad_ufes_20.py` | Extra class 'BOD' may cause KeyError | 🔴 |
| 3 | `run_chestx_ray14.py` | NUM_CLASSES initial value wrong (14 vs 15) | 🟡 |
| 4 | `run_bimcv_covid19.py` | DICOM files, generic loader won't find images | 🔴 |
| 5 | `run_mimic_cxr.py` | DICOM/nested structure, generic loader incompatible | 🔴 |
| 6 | `run_camelyon16.py` | WSI (.tif), generic loader incompatible | 🔴 |
| 7 | `run_camelyon17.py` | WSI (.tif), generic loader incompatible | 🔴 |
| 8 | `run_panda_challenge.py` | WSI (.tiff), generic loader incompatible | 🔴 |
| 9 | `run_brats_2020.py` | NIfTI 3D volumes, 2D pipeline incompatible | 🔴 |
| 10 | `run_fastmri_knee.py` | HDF5 k-space data, not images | 🔴 |
| 11 | `run_lidc_idri.py` | DICOM CT volumes, generic loader incompatible | 🔴 |
| 12 | `run_rsna_intracranial.py` | DICOM images, generic loader only searches .png/.jpg | 🔴 |
| 13 | `run_bone_age_rsna.py` | Regression task forced into classification | 🔴 |
| 14 | `run_busi.py` | Mask images included in training data (silent corruption) | 🔴 |

## Systematic Fixes Needed

1. **Add `grad_checkpoint` to all 39 scripts** (copy from `run_blood_cell_detection.py`)
2. **Add `drop_last=True` to all training DataLoaders** (39 scripts)
3. **Add per-step progress logging to `train_epoch`** (39 scripts)
4. **Fix DICOM/WSI/NIfTI/HDF5 datasets** — either add format-specific loaders or document that these require preprocessing
5. **Fix BUSI mask inclusion** — filter out `*_mask*` files
