# Medical Imaging Dataset Registry & Detailed Reference

This document serves as a centralized, detailed registry of all 40 medical imaging datasets integrated into the STTP-Net Imbalance Benchmark suite. It covers acronyms/full forms, imaging modalities, domain groupings, class counts, sample sizes, imbalance ratios, and detailed descriptions of all class labels.

---

## 1. Quick Reference & Comparative Summary

| # | Dataset | Full Name | Domain | Modality | Classes | Samples | Imbalance Ratio | Distribution |
|---|---|---|---|---|---|---|---|---|
| 1 | **HAM10000** | Human Against Machine 10,000 | Dermoscopy | Dermoscopy | 7 | 10,015 | 60.0 | Highly Imbalanced |
| 2 | **ISIC_2019** | International Skin Imaging Collaboration 2019 | Dermoscopy | Dermoscopy | 8 | 25,331 | 50.0 | Imbalanced |
| 3 | **ISIC_2020** | International Skin Imaging Collaboration 2020 | Dermoscopy | Dermoscopy | 2 | 33,126 | 98.0 | Highly Imbalanced |
| 4 | **PAD_UFES_20** | Dermatological and Clinical Dataset (Univ. of Espírito Santo) | Dermoscopy | Clinical | 6 | 2,298 | 20.0 | Imbalanced |
| 5 | **DDI** | Diverse Dermatology Images | Dermoscopy | Clinical | 78 | 656 | 100.0 | Extremely Imbalanced |
| 6 | **ChestX-ray14** | NIH Chest X-ray 14 | Radiography | X-ray | 14 | 112,120 | 30.0 | Imbalanced |
| 7 | **CheXpert** | Stanford Chest eXpert | Radiography | X-ray | 14 | 224,316 | 25.0 | Imbalanced |
| 8 | **COVID-19_Radiography** | COVID-19 Radiography Database | Radiography | X-ray | 4 | 21,165 | 10.0 | Imbalanced |
| 9 | **RSNA_Pneumonia** | RSNA Pneumonia Detection Challenge | Radiography | X-ray | 2 | 26,684 | 1.5 | Balanced |
| 10 | **BIMCV_COVID19** | BIMCV COVID19+ (Valencian Region Medical Image Bank) | Radiography | X-ray | 3 | 23,000 | 15.0 | Imbalanced |
| 11 | **TB_Chest_Xray** | Tuberculosis Chest X-ray Dataset | Radiography | X-ray | 2 | 4,200 | 4.0 | Imbalanced |
| 12 | **MIMIC-CXR** | Medical Information Mart for Intensive Care Chest X-ray | Radiography | X-ray | 14 | 377,110 | 50.0 | Highly Imbalanced |
| 13 | **VinDr-CXR** | VinBigData Chest X-ray Abnormalities Detection | Radiography | X-ray | 22 | 18,000 | 60.0 | Highly Imbalanced |
| 14 | **BreakHis** | Breast Cancer Histopathological Database | Histopathology | Microscopy | 8 | 7,909 | 3.0 | Imbalanced |
| 15 | **PCam** | PatchCamelyon | Histopathology | Microscopy | 2 | 327,680 | 1.0 | Balanced |
| 16 | **NCT-CRC-HE-100K** | NCT Colorectal Cancer Histology HE 100K | Histopathology | Microscopy | 9 | 100,000 | 1.2 | Balanced |
| 17 | **Camelyon16** | Cancer Metastasis Detection in Lymph Nodes 2016 Challenge | Histopathology | WSI (TIFF) | 2 | 400 | 1.0 | Balanced |
| 18 | **Camelyon17** | Cancer Metastasis Detection in Lymph Nodes 2017 Challenge | Histopathology | WSI (TIFF) | 2 | 1,000 | 2.0 | Imbalanced |
| 19 | **PANDA_Challenge** | Prostate Cancer Grade Assessment Challenge | Histopathology | WSI (TIFF) | 6 | 10,616 | 4.0 | Imbalanced |
| 20 | **APTOS_2019** | Asia Pacific Tele-Ophthalmology Society 2019 | Retinal | Fundus | 5 | 3,662 | 10.0 | Highly Imbalanced |
| 21 | **EyePACS** | Diabetic Retinopathy Detection (EyePACS) | Retinal | Fundus | 5 | 35,126 | 30.0 | Highly Imbalanced |
| 22 | **MESSIDOR-2** | Messidor-2 (Optical Retinology Evaluation Dataset) | Retinal | Fundus | 5 | 1,748 | 8.0 | Imbalanced |
| 23 | **STARE** | Structured Analysis of the Retina | Retinal | Fundus | 2 | 400 | 5.0 | Imbalanced |
| 24 | **IDRiD** | Indian Diabetic Retinopathy Image Dataset | Retinal | Fundus | 4 | 516 | 6.0 | Imbalanced |
| 25 | **ODIR-5K** | Ocular Disease Intelligent Recognition 5K | Retinal | Fundus | 8 | 5,000 | 12.0 | Imbalanced |
| 26 | **BraTS_2020** | Brain Tumor Segmentation Challenge 2020 | Neuroimaging | MRI (3D) | 4 | 369 | 3.0 | Imbalanced |
| 27 | **OASIS** | Open Access Series of Imaging Studies | Neuroimaging | MRI (2D/3D)| 4 | 416 | 2.5 | Imbalanced |
| 28 | **FastMRI_Knee** | NYU fastMRI Knee Dataset | Musculoskeletal | MRI (HDF5) | 2 | 10,000 | 4.0 | Imbalanced |
| 29 | **CT_Medical_Images** | SIIM Medical Image Analysis Tutorial CT | Oncology | CT | 2 | 475 | 2.0 | Imbalanced |
| 30 | **LIDC-IDRI** | Lung Image Database Consortium / TCIA | Oncology | CT (DICOM) | 4 | 1,018 | 5.0 | Imbalanced |
| 31 | **RSNA_Intracranial** | RSNA Intracranial Hemorrhage Detection | Neuroimaging | CT (DICOM) | 6 | 874,036 | 20.0 | Highly Imbalanced |
| 32 | **Kvasir** | Kvasir GI Endoscopy Dataset | Endoscopy | Endoscopy | 8 | 8,000 | 1.0 | Balanced |
| 33 | **HyperKvasir** | Simula HyperKvasir Endoscopy Dataset | Endoscopy | Endoscopy | 23 | 110,079 | 50.0 | Highly Imbalanced |
| 34 | **BUSI** | Breast Ultrasound Images Dataset | Oncology | Ultrasound | 3 | 780 | 3.0 | Imbalanced |
| 35 | **Cervical_Cancer_Intel**| Intel & MobileODT Cervical Cancer Screening | Oncology | Clinical | 3 | 8,222 | 4.0 | Imbalanced |
| 36 | **PolypGen** | PolypGen Dataset | Endoscopy | Endoscopy | 2 | 3,142 | 2.0 | Imbalanced |
| 37 | **Blood_Cell_Detection**| WBC Classification (Paul Mooney) | Hematology | Microscopy | 4 | 12,500 | 2.0 | Imbalanced |
| 38 | **Malaria_Cell** | NIH Malaria Cell Images | Hematology | Microscopy | 2 | 27,558 | 1.0 | Balanced |
| 39 | **Bone_Age_RSNA** | RSNA Pediatric Bone Age | Musculoskeletal | X-ray | 2 | 12,611 | 5.0 | Imbalanced |
| 40 | **MURA** | Musculoskeletal Radiographs (Stanford) | Musculoskeletal | X-ray | 2 | 40,561 | 2.0 | Imbalanced |

---

## 2. Detailed Dataset Profiles

### 1. HAM10000
*   **Full Name**: Human Against Machine 10,000
*   **Modality & Domain**: Dermoscopy | Dermatology
*   **Samples & Classes**: 10,015 images | 7 classes
*   **Imbalance Ratio**: 60.0 (Highly Imbalanced)
*   **Class Labels**:
    *   `nv` (Melanocytic nevi) - Common benign mole. (Heavily dominates dataset).
    *   `mel` (Melanoma) - Deadly malignant skin cancer.
    *   `bcc` (Basal cell carcinoma) - Common malignant epithelial skin cancer.
    *   `akiec` (Actinic keratoses and intraepithelial carcinoma) - Pre-cancerous lesions.
    *   `bkl` (Benign keratosis-like lesions) - Non-cancerous solar lentigines/seborrheic keratoses.
    *   `df` (Dermatofibroma) - Benign skin nodule.
    *   `vasc` (Vascular lesions) - Angiomas, angiokeratomas, pyogenic granulomas.

### 2. ISIC_2019
*   **Full Name**: International Skin Imaging Collaboration 2019 Challenge
*   **Modality & Domain**: Dermoscopy | Dermatology
*   **Samples & Classes**: 25,331 images | 8 classes
*   **Imbalance Ratio**: 50.0 (Imbalanced)
*   **Class Labels**:
    *   `MEL` (Melanoma)
    *   `NV` (Melanocytic nevus)
    *   `BCC` (Basal cell carcinoma)
    *   `AK` (Actinic keratosis)
    *   `BKL` (Benign keratosis)
    *   `DF` (Dermatofibroma)
    *   `VASC` (Vascular lesion)
    *   `SCC` (Squamous cell carcinoma) - Invasive malignant skin cancer.

### 3. ISIC_2020
*   **Full Name**: International Skin Imaging Collaboration 2020 Challenge
*   **Modality & Domain**: Dermoscopy | Dermatology
*   **Samples & Classes**: 33,126 images | 2 classes (Binary classification)
*   **Imbalance Ratio**: 98.0 (Highly Imbalanced - extreme majority are benign)
*   **Class Labels**:
    *   `0`: Benign lesions
    *   `1`: Malignant melanoma

### 4. PAD_UFES_20
*   **Full Name**: Dermatological and Clinical Dataset from Federal University of Espírito Santo
*   **Modality & Domain**: Clinical Images (Smartphone/DSLR) | Dermatology
*   **Samples & Classes**: 2,298 images | 6 or 7 classes
*   **Imbalance Ratio**: 20.0 (Imbalanced)
*   **Class Labels**:
    *   `BCC` (Basal Cell Carcinoma)
    *   `SCC` (Squamous Cell Carcinoma)
    *   `ACK` (Actinic Keratosis)
    *   `SEK` (Seborrheic Keratosis)
    *   `BOD` (Bowen’s Disease - Squamous cell carcinoma in situ)
    *   `MEL` (Melanoma)
    *   `NEV` (Nevus)

### 5. DDI
*   **Full Name**: Diverse Dermatology Images
*   **Modality & Domain**: Clinical Images | Dermatology (specifically curated to contain skin types I-VI)
*   **Samples & Classes**: 656 images | 78 classes
*   **Imbalance Ratio**: 100.0 (Extremely Imbalanced due to few shots per disease class)
*   **Class Labels**: Contains 78 distinct disease labels spanning inflammatory, neoplastic, and infectious skin pathologies.

### 6. ChestX-ray14
*   **Full Name**: NIH Chest X-ray 14
*   **Modality & Domain**: X-ray | Chest Radiography
*   **Samples & Classes**: 112,120 images | 14 finding classes (+ 'No Finding')
*   **Imbalance Ratio**: 30.0 (Imbalanced)
*   **Class Labels**:
    *   `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`
    *   *Note*: Multi-label in nature, but the pipeline simplifies this task to single-label by mapping images to their **first** listed finding.

### 7. CheXpert
*   **Full Name**: Stanford Chest eXpert
*   **Modality & Domain**: X-ray | Chest Radiography
*   **Samples & Classes**: 224,316 images | 14 classes
*   **Imbalance Ratio**: 25.0 (Imbalanced)
*   **Class Labels**:
    *   Includes observation categories: `No Finding`, `Enlarged Cardiomediastinum`, `Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`, `Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`, `Fracture`, `Support Devices`.

### 8. COVID-19_Radiography
*   **Full Name**: COVID-19 Radiography Database
*   **Modality & Domain**: X-ray | Chest Radiography
*   **Samples & Classes**: 21,165 images | 4 classes
*   **Imbalance Ratio**: 10.0 (Imbalanced)
*   **Class Labels**:
    *   `COVID` (COVID-19 positive)
    *   `Normal` (Healthy control)
    *   `Viral Pneumonia` (Non-COVID viral infection)
    *   `Lung Opacity` (Other non-specific opacity findings)

### 9. RSNA_Pneumonia
*   **Full Name**: RSNA Pneumonia Detection Challenge
*   **Modality & Domain**: X-ray (DICOM format converted to RGB) | Chest Radiography
*   **Samples & Classes**: 26,684 images | 2 classes (Binary)
*   **Imbalance Ratio**: 1.5 (Balanced)
*   **Class Labels**:
    *   `Normal/Non-Pneumonia` (No pathological opacities)
    *   `Pneumonia` (Visible lung opacities indicative of infection)

### 10. BIMCV_COVID19
*   **Full Name**: BIMCV COVID19+ (Valencian Region Medical Image Bank)
*   **Modality & Domain**: X-ray / CT slices | Chest Radiography
*   **Samples & Classes**: 23,000 images | 3 classes
*   **Imbalance Ratio**: 15.0 (Imbalanced)
*   **Class Labels**:
    *   `COVID-19` (Positive findings)
    *   `Non-COVID Pneumonia/Pathology` (Other diseases)
    *   `Normal` (Healthy chest)

### 11. TB_Chest_Xray
*   **Full Name**: Tuberculosis Chest X-ray Dataset
*   **Modality & Domain**: X-ray | Chest Radiography
*   **Samples & Classes**: 4,200 images | 2 classes (Binary)
*   **Imbalance Ratio**: 4.0 (Imbalanced)
*   **Class Labels**:
    *   `Tuberculosis` (TB infected chest radiographs)
    *   `Normal` (Healthy chest radiographs)

### 12. MIMIC-CXR
*   **Full Name**: Medical Information Mart for Intensive Care Chest X-ray
*   **Modality & Domain**: X-ray | Chest Radiography
*   **Samples & Classes**: 377,110 images | 14 classes
*   **Imbalance Ratio**: 50.0 (Highly Imbalanced due to rare findings)
*   **Class Labels**: Same 14 finding classes as CheXpert (Atelectasis, Cardiomegaly, Consolidation, etc.).

### 13. VinDr-CXR
*   **Full Name**: VinBigData Chest X-ray Abnormalities Detection
*   **Modality & Domain**: X-ray (DICOM format) | Chest Radiography
*   **Samples & Classes**: 18,000 images | 22 classes
*   **Imbalance Ratio**: 60.0 (Highly Imbalanced)
*   **Class Labels**: 22 specific categories of clinical findings and structures (e.g., Aortic enlargement, Cardiomegaly, Pleural effusion, etc.).

### 14. BreakHis
*   **Full Name**: Breast Cancer Histopathology Dataset
*   **Modality & Domain**: H&E Stained Microscopy | Pathology
*   **Samples & Classes**: 7,909 images | 8 classes (4 benign types, 4 malignant types)
*   **Imbalance Ratio**: 3.0 (Imbalanced)
*   **Class Labels**:
    *   *Benign*: `adenosis`, `fibroadenoma`, `phyllodes_tumor`, `tubular_adenoma`
    *   *Malignant*: `ductal_carcinoma`, `lobular_carcinoma`, `mucinous_carcinoma`, `papillary_carcinoma`

### 15. PCam
*   **Full Name**: PatchCamelyon
*   **Modality & Domain**: Histopathology Microscopy (96x96 pixels) | Pathology
*   **Samples & Classes**: 327,680 images | 2 classes (Binary)
*   **Imbalance Ratio**: 1.0 (Perfect 50/50 balance)
*   **Class Labels**:
    *   `0`: No metastatic tumor cells present in the center 32x32px region
    *   `1`: Metastatic tumor cells present in the center 32x32px region

### 16. NCT-CRC-HE-100K
*   **Full Name**: NCT Colorectal Cancer Histology HE 100K
*   **Modality & Domain**: H&E Stained Microscopy | Gastroenterology/Pathology
*   **Samples & Classes**: 100,000 images | 9 classes
*   **Imbalance Ratio**: 1.2 (Balanced)
*   **Class Labels**:
    *   `ADI` (Adipose tissue / fat)
    *   `DEB` (Debris / necrotic tissue)
    *   `LYM` (Lymphocytes / white blood cells)
    *   `MUC` (Mucus secretion regions)
    *   `MUS` (Smooth muscle tissue)
    *   `NORM` (Normal colon mucosa)
    *   `STR` (Cancer-associated stroma / connective tissue)
    *   `TUM` (Colorectal adenocarcinoma epithelium / active tumor cells)
    *   `BACK` (Background / empty slide parts)

### 17. Camelyon16
*   **Full Name**: Cancer Metastasis Detection in Lymph Nodes 2016 Challenge
*   **Modality & Domain**: Whole Slide Images (WSI) | Pathology
*   **Samples & Classes**: 400 WSIs (requires preprocessing/patch extraction) | 2 classes
*   **Imbalance Ratio**: 1.0 (Balanced)
*   **Class Labels**:
    *   `normal`: Lymph node slide without metastasis
    *   `tumor`: Lymph node slide with metastasis regions

### 18. Camelyon17
*   **Full Name**: Cancer Metastasis Detection in Lymph Nodes 2017 Challenge
*   **Modality & Domain**: Whole Slide Images (WSI) | Pathology
*   **Samples & Classes**: 1,000 WSIs (requires preprocessing/patch extraction) | 2 classes
*   **Imbalance Ratio**: 2.0 (Imbalanced)
*   **Class Labels**:
    *   `normal`: No metastasis detected
    *   `tumor`: Metastatic disease present in the lymph nodes

### 19. PANDA_Challenge
*   **Full Name**: Prostate Cancer Grade Assessment Challenge
*   **Modality & Domain**: Whole Slide Images (WSI) | Urology/Pathology
*   **Samples & Classes**: 10,616 WSIs | 6 classes (Severity progression)
*   **Imbalance Ratio**: 4.0 (Imbalanced)
*   **Class Labels**: ISUP grade classification:
    *   `0`: Benign prostate tissue (healthy)
    *   `1`: Gleason score 6 (low-grade cancer)
    *   `2`: Gleason score 3+4=7 (intermediate risk)
    *   `3`: Gleason score 4+3=7 (intermediate risk, worse prognosis)
    *   `4`: Gleason score 8 (high-grade cancer)
    *   `5`: Gleason score 9 or 10 (extremely advanced cancer)

### 20. APTOS_2019
*   **Full Name**: Asia Pacific Tele-Ophthalmology Society 2019 Diabetic Retinopathy
*   **Modality & Domain**: Color Fundus Photography | Ophthalmology
*   **Samples & Classes**: 3,662 images | 5 classes (Graded severity)
*   **Imbalance Ratio**: 10.0 (Highly Imbalanced)
*   **Class Labels**:
    *   `0`: No Diabetic Retinopathy (DR)
    *   `1`: Mild Non-proliferative DR
    *   `2`: Moderate Non-proliferative DR
    *   `3`: Severe Non-proliferative DR
    *   `4`: Proliferative DR

### 21. EyePACS
*   **Full Name**: Diabetic Retinopathy Detection (EyePACS Challenge)
*   **Modality & Domain**: Color Fundus Photography | Ophthalmology
*   **Samples & Classes**: 35,126 images | 5 classes
*   **Imbalance Ratio**: 30.0 (Highly Imbalanced - majority are grade 0)
*   **Class Labels**: Graded severity mapping from `0` (No DR) to `4` (Proliferative DR), identical to the APTOS specification.

### 22. MESSIDOR-2
*   **Full Name**: Messidor-2 Diabetic Retinopathy Dataset
*   **Modality & Domain**: Color Fundus Photography | Ophthalmology
*   **Samples & Classes**: 1,748 images | 5 classes
*   **Imbalance Ratio**: 8.0 (Imbalanced)
*   **Class Labels**: Diabetic retinopathy progression levels `0` to `4`.

### 23. STARE
*   **Full Name**: Structured Analysis of the Retina
*   **Modality & Domain**: Color Fundus Photography | Ophthalmology
*   **Samples & Classes**: 400 images | 2 classes (Binary)
*   **Imbalance Ratio**: 5.0 (Imbalanced)
*   **Class Labels**:
    *   `normal`: Healthy retina
    *   `abnormal`: Pathological changes detected (e.g. vascular occlusion, hemorrhages)

### 24. IDRiD
*   **Full Name**: Indian Diabetic Retinopathy Image Dataset
*   **Modality & Domain**: Color Fundus Photography | Ophthalmology
*   **Samples & Classes**: 516 images | 4 classes
*   **Imbalance Ratio**: 6.0 (Imbalanced)
*   **Class Labels**: Retinopathy clinical staging levels.

### 25. ODIR-5K
*   **Full Name**: Ocular Disease Intelligent Recognition 5K
*   **Modality & Domain**: Color Fundus Photography (Left & Right eye pairs) | Ophthalmology
*   **Samples & Classes**: 5,000 images | 8 classes
*   **Imbalance Ratio**: 12.0 (Imbalanced)
*   **Class Labels**:
    *   `Normal` (N)
    *   `Diabetes` (D)
    *   `Glaucoma` (G)
    *   `Cataract` (C)
    *   `AMD` (Age-related macular degeneration) (A)
    *   `Hypertension` (H)
    *   `Myopia` (M)
    *   `Other diseases` (O)

### 26. BraTS_2020
*   **Full Name**: Brain Tumor Segmentation Challenge 2020
*   **Modality & Domain**: 3D MRI Volumes (T1, T1c, T2, FLAIR) | Neurology/Neuro-Oncology
*   **Samples & Classes**: 369 MRI cases (requires volumetric slice extraction) | 4 classes
*   **Imbalance Ratio**: 3.0 (Imbalanced)
*   **Class Labels**: Segmentation classes (simplified to slice-level or voxel-level representation):
    *   `0`: Background (Healthy tissue)
    *   `1`: Necrotic and Non-enhancing Tumor Core (NCR/NET)
    *   `2`: Peritumoral Edema (ED)
    *   `3`: GD-enhancing Tumor (ET)

### 27. OASIS
*   **Full Name**: Open Access Series of Imaging Studies
*   **Modality & Domain**: MRI (Brain slices) | Neurology/Geriatrics
*   **Samples & Classes**: 416 cases | 4 classes
*   **Imbalance Ratio**: 2.5 (Imbalanced)
*   **Class Labels**: Clinical Dementia Rating (CDR):
    *   `0`: No cognitive impairment (Normal)
    *   `0.5`: Very mild dementia
    *   `1`: Mild dementia
    *   `2`: Moderate dementia

### 28. FastMRI_Knee
*   **Full Name**: NYU fastMRI Knee Dataset
*   **Modality & Domain**: MRI reconstructed slices (HDF5 raw k-space data) | Orthopedics
*   **Samples & Classes**: 10,000 slices | 2 classes (Binary)
*   **Imbalance Ratio**: 4.0 (Imbalanced)
*   **Class Labels**:
    *   `0`: Control group / No abnormality detected
    *   `1`: Abnormality present (e.g. meniscus tear, ligament tear)

### 29. CT_Medical_Images
*   **Full Name**: SIIM Medical Image Analysis Tutorial CT Dataset
*   **Modality & Domain**: Computed Tomography (CT) | Oncology/Body region
*   **Samples & Classes**: 475 images | 2 classes (Binary)
*   **Imbalance Ratio**: 2.0 (Imbalanced)
*   **Class Labels**: Body part scan classification:
    *   `0`: Chest scans
    *   `1`: Abdomen/other region scans

### 30. LIDC-IDRI
*   **Full Name**: Lung Image Database Consortium and Image Database Resource Initiative
*   **Modality & Domain**: CT scans (Thoracic DICOM slices) | Pulmonology/Oncology
*   **Samples & Classes**: 1,018 cases | 4 classes (Grouped risk categories)
*   **Imbalance Ratio**: 5.0 (Imbalanced)
*   **Class Labels**: Nodule malignancy score group:
    *   `1`: Highly unlikely malignant
    *   `2`: Indeterminate
    *   `3`: Suspicious / Moderately likely
    *   `4`: Highly suspicious of cancer

### 31. RSNA_Intracranial
*   **Full Name**: RSNA Intracranial Hemorrhage Detection Challenge
*   **Modality & Domain**: CT scans (Brain DICOM slices) | Neurology
*   **Samples & Classes**: 874,036 images | 6 classes (Subtypes)
*   **Imbalance Ratio**: 20.0 (Highly Imbalanced)
*   **Class Labels**: Hemorrhage subtype classification:
    *   `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`, `any` (if any hemorrhage is present).

### 32. Kvasir
*   **Full Name**: Kvasir Gastroenterology Endoscopy Dataset
*   **Modality & Domain**: Gastroscopy/Colonoscopy Images | Gastroenterology
*   **Samples & Classes**: 8,000 images | 8 classes (Equal split of 1,000 images per class)
*   **Imbalance Ratio**: 1.0 (Balanced)
*   **Class Labels**:
    *   `dyed-barretts-esophagus` (Premalignant esophageal lesion)
    *   `dyed-resection-margins` (Polyp removal margins)
    *   `esophagitis` (Inflammation of the esophagus)
    *   `normal-cecum` (Anatomical landmark in the colon)
    *   `normal-pylorus` (Anatomical boundary between stomach and duodenum)
    *   `normal-z-line` (Anatomical transition zone in esophagus)
    *   `polyps` (Pre-cancerous mucosal growths)
    *   `ulcerative-colitis` (Chronic inflammatory bowel disease lesions)

### 33. HyperKvasir
*   **Full Name**: Simula HyperKvasir Endoscopy Dataset
*   **Modality & Domain**: Gastroscopy/Colonoscopy Images | Gastroenterology
*   **Samples & Classes**: 110,079 images | 23 classes
*   **Imbalance Ratio**: 50.0 (Highly Imbalanced - anatomical landmarks dominate pathological findings)
*   **Class Labels**: 23 categories dividing findings into:
    *   *Pathological*: Polyps, Esophagitis, Ulcers, Barrett's.
    *   *Anatomical*: Cecum, Pylorus, Z-line, retroflexed views.
    *   *Interventions*: Dyed margins, clean resection.

### 34. BUSI
*   **Full Name**: Breast Ultrasound Images Dataset
*   **Modality & Domain**: Ultrasound | Oncology
*   **Samples & Classes**: 780 images | 3 classes
*   **Imbalance Ratio**: 3.0 (Imbalanced)
*   **Class Labels**:
    *   `benign`: Non-cancerous breast tumors (fibroadenomas, cysts)
    *   `malignant`: Breast cancer lesions
    *   `normal`: Healthy breast tissue without lumps

### 35. Cervical_Cancer_Intel
*   **Full Name**: Intel & MobileODT Cervical Cancer Screening Challenge
*   **Modality & Domain**: Clinical Speculum Photography | Gynecology/Oncology
*   **Samples & Classes**: 8,222 images | 3 classes
*   **Imbalance Ratio**: 4.0 (Imbalanced)
*   **Class Labels**: Cervix type categories based on transformational zone visibility:
    *   `Type_1`: Completely ectocervical (highly visible, easier to inspect)
    *   `Type_2`: Partially endocervical (partially hidden)
    *   `Type_3`: Completely endocervical (deep in the canal, hardest to evaluate)

### 36. PolypGen
*   **Full Name**: PolypGen Dataset
*   **Modality & Domain**: Colonoscopy Images | Gastroenterology
*   **Samples & Classes**: 3,142 images | 2 classes (Binary)
*   **Imbalance Ratio**: 2.0 (Imbalanced)
*   **Class Labels**:
    *   `no polyp`: Normal colon mucosa
    *   `polyp`: Colon mucosal polyp present

### 37. Blood_Cell_Detection
*   **Full Name**: Blood Cell Images (White Blood Cell Classification)
*   **Modality & Domain**: Peripheral Blood Smear Microscopy | Hematology
*   **Samples & Classes**: 12,500 images | 4 classes
*   **Imbalance Ratio**: 2.0 (Imbalanced)
*   **Class Labels**: WBC cell types:
    *   `EOSINOPHIL`: Immune cells reacting to allergies/parasites
    *   `LYMPHOCYTE`: Vital cells for viral response/antibody production (B/T cells)
    *   `MONOCYTE`: Largest WBCs, phagocytize foreign pathogens
    *   `NEUTROPHIL`: Most common immune cell, responds to bacterial infections

### 38. Malaria_Cell
*   **Full Name**: NIH Malaria Cell Images
*   **Modality & Domain**: Thin Blood Smear Microscopy | Hematology/Infectious Disease
*   **Samples & Classes**: 27,558 images | 2 classes (Binary, 50/50 balance)
*   **Imbalance Ratio**: 1.0 (Balanced)
*   **Class Labels**:
    *   `Parasitized`: Red blood cells infected with Plasmodium falciparum malaria parasite
    *   `Uninfected`: Normal, healthy red blood cells

### 39. Bone_Age_RSNA
*   **Full Name**: RSNA Pediatric Bone Age Dataset
*   **Modality & Domain**: Radiography | Musculoskeletal/Pediatrics
*   **Samples & Classes**: 12,611 images | 2 classes (Binary classification)
*   **Imbalance Ratio**: 5.0 (Imbalanced)
*   **Class Labels**: Sex-based classification (though the original competition is a continuous regression of age in months):
    *   `0`: Female
    *   `1`: Male

### 40. MURA
*   **Full Name**: Musculoskeletal Radiographs (Stanford MURA Challenge)
*   **Modality & Domain**: X-ray | Musculoskeletal
*   **Samples & Classes**: 40,561 images | 2 classes (Binary)
*   **Imbalance Ratio**: 2.0 (Imbalanced)
*   **Class Labels**:
    *   `positive`: Joint/bone radiograph showing abnormalities (fractures, arthritis, hardware/screws)
    *   `negative`: Healthy joint/bone radiograph
