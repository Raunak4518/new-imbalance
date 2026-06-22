"""
One-time patcher: Replace the broken Kaggle API download logic in every script
with dataset-specific direct wget downloads from original sources.
"""
import os, re

SCRIPTS_DIR = r"d:\new imbalance\scripts"

# ── The OLD download block that exists in every script (lines 64-126 pattern) ──
# We match from "# ── 1. DOWNLOAD" to right before "# ── 2. DATA LOADING"
OLD_DOWNLOAD_PATTERN = re.compile(
    r'# ── 1\. (?:DOWNLOAD|DISCOVER).*?(?=# ── 2\. DATA LOADING)',
    re.DOTALL
)

# ── Dataset-specific download configurations ──
# Category A: Direct public URL available
# Category B: Kaggle-only (must use Add Data)
# Category C: Requires registration

DOWNLOAD_CONFIGS = {
    # ── ISIC_2019: Already fixed manually, skip ──
    "run_isic_2019.py": None,

    # ── Category A: Direct public URL downloads ──
    "run_isic_2020.py": {
        "required_file": "ISIC_2020_Training_GroundTruth_v2.csv",
        "urls": {
            "ISIC_2020_Training_GroundTruth_v2.csv": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv",
            "ISIC_2020_Training_JPEG.zip": "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip",
        },
        "kaggle_handle": "cdeotte/jpeg-melanoma-256x256",
    },
    "run_malaria_cell.py": {
        "required_file": "Parasitized",  # folder name
        "urls": {
            "cell_images.zip": "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
        },
        "kaggle_handle": "iarunava/cell-images-for-detecting-malaria",
    },
    "run_kvasir.py": {
        "required_file": "dyed-lifted-polyps",  # one of the 8 class folders
        "urls": {
            "kvasir-dataset-v2.zip": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
        },
        "kaggle_handle": "meetnagadia/kvasir-dataset",
    },
    "run_hyperkvasir.py": {
        "required_file": "labeled-images",
        "urls": {
            "hyper-kvasir-labeled-images.zip": "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip",
        },
        "kaggle_handle": "N/A",
    },
    "run_nct_crc_he_100k.py": {
        "required_file": "ADI",  # one of the 9 class folders
        "urls": {
            "NCT-CRC-HE-100K.zip": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip",
        },
        "kaggle_handle": "kmader/colorectal-histology-mnist",
    },

    # ── Category B: Kaggle-only datasets (must attach via Add Data) ──
    "run_ham10000.py": {
        "required_file": "HAM10000_metadata.csv",
        "urls": {},
        "kaggle_handle": "kmader/skin-cancer-mnist-ham10000",
    },
    "run_chestx_ray14.py": {
        "required_file": "Data_Entry_2017.csv",
        "urls": {},
        "kaggle_handle": "nih-chest-xrays/data",
    },
    "run_covid_19_radiography.py": {
        "required_file": "COVID",  # class folder
        "urls": {},
        "kaggle_handle": "tawsifurrahman/covid19-radiography-database",
    },
    "run_tb_chest_xray.py": {
        "required_file": "Tuberculosis",  # class folder
        "urls": {},
        "kaggle_handle": "tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
    },
    "run_breakhis.py": {
        "required_file": "BreaKHis_v1",
        "urls": {},
        "kaggle_handle": "ambarish/breakhis",
    },
    "run_odir_5k.py": {
        "required_file": "ODIR-5K",
        "urls": {},
        "kaggle_handle": "andrewmvd/ocular-disease-recognition-odir5k",
    },
    "run_brats_2020.py": {
        "required_file": "BraTS20",
        "urls": {},
        "kaggle_handle": "awsaf49/brats20-dataset-training-validation",
    },
    "run_oasis.py": {
        "required_file": "oasis",
        "urls": {},
        "kaggle_handle": "jboysen/mri-and-alzheimers",
    },
    "run_ct_medical_images.py": {
        "required_file": "overview.csv",
        "urls": {},
        "kaggle_handle": "kmader/siim-medical-image-analysis-tutorial",
    },
    "run_busi.py": {
        "required_file": "benign",  # class folder
        "urls": {},
        "kaggle_handle": "aryashah2k/breast-ultrasound-images-dataset",
    },
    "run_polypgen.py": {
        "required_file": "polyp",
        "urls": {},
        "kaggle_handle": "debeshjha/polypgen",
    },
    "run_bone_age_rsna.py": {
        "required_file": "boneage-training-dataset",
        "urls": {},
        "kaggle_handle": "kmader/rsna-bone-age",
    },
    "run_blood_cell_detection.py": {
        "required_file": "TRAIN",
        "urls": {},
        "kaggle_handle": "paultimothymooney/blood-cells",
    },
    # Kaggle Competitions
    "run_rsna_pneumonia.py": {
        "required_file": "stage_2_train_labels.csv",
        "urls": {},
        "kaggle_handle": "rsna-pneumonia-detection-challenge",
        "is_competition": True,
    },
    "run_vindr_cxr.py": {
        "required_file": "train.csv",
        "urls": {},
        "kaggle_handle": "vinbigdata-chest-xray-abnormalities-detection",
        "is_competition": True,
    },
    "run_pcam.py": {
        "required_file": "train_labels.csv",
        "urls": {},
        "kaggle_handle": "histopathologic-cancer-detection",
        "is_competition": True,
    },
    "run_panda_challenge.py": {
        "required_file": "train.csv",
        "urls": {},
        "kaggle_handle": "prostate-cancer-grade-assessment",
        "is_competition": True,
    },
    "run_aptos_2019.py": {
        "required_file": "train.csv",
        "urls": {},
        "kaggle_handle": "aptos2019-blindness-detection",
        "is_competition": True,
    },
    "run_eyepacs.py": {
        "required_file": "trainLabels.csv",
        "urls": {},
        "kaggle_handle": "diabetic-retinopathy-detection",
        "is_competition": True,
    },
    "run_rsna_intracranial.py": {
        "required_file": "stage_2_train.csv",
        "urls": {},
        "kaggle_handle": "rsna-intracranial-hemorrhage-detection",
        "is_competition": True,
    },
    "run_cervical_cancer_intel.py": {
        "required_file": "train",  # folder
        "urls": {},
        "kaggle_handle": "intel-mobileodt-cervical-cancer-screening",
        "is_competition": True,
    },

    # ── Category C: Requires registration/credentials ──
    "run_chexpert.py": {
        "required_file": "train.csv",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "manual_note": "Register at Stanford ML Group website to download CheXpert",
    },
    "run_mimic_cxr.py": {
        "required_file": "mimic-cxr",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://physionet.org/content/mimic-cxr-jpg/2.0.0/",
        "manual_note": "Requires PhysioNet credentialed access",
    },
    "run_bimcv_covid19.py": {
        "required_file": "bimcv",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/",
        "manual_note": "Register at BIMCV portal to download",
    },
    "run_pad_ufes_20.py": {
        "required_file": "metadata.csv",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        "manual_note": "Download from Mendeley Data (free, no auth required, but redirect-based)",
    },
    "run_ddi.py": {
        "required_file": "ddi",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://ddi-dataset.github.io/",
        "manual_note": "Request access from DDI project page",
    },
    "run_camelyon16.py": {
        "required_file": "camelyon",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://camelyon16.grand-challenge.org/Data/",
        "manual_note": "Register at Grand Challenge to download",
    },
    "run_camelyon17.py": {
        "required_file": "camelyon",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://camelyon17.grand-challenge.org/Data/",
        "manual_note": "Register at Grand Challenge to download",
    },
    "run_messidor_2.py": {
        "required_file": "messidor",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "http://www.adcis.net/en/third-party/messidor/",
        "manual_note": "Register at ADCIS to download MESSIDOR-2",
    },
    "run_stare.py": {
        "required_file": "stare",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://cecas.clemson.edu/~ahoover/stare/",
        "manual_note": "Download individual images from Clemson STARE page",
    },
    "run_idrid.py": {
        "required_file": "IDRiD",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
        "manual_note": "Download from IEEE DataPort (free, requires IEEE login)",
    },
    "run_fastmri_knee.py": {
        "required_file": "fastmri",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://fastmri.med.nyu.edu/",
        "manual_note": "Register at NYU FastMRI portal to download",
    },
    "run_lidc_idri.py": {
        "required_file": "LIDC",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
        "manual_note": "Download via TCIA NBIA Data Retriever",
    },
    "run_mura.py": {
        "required_file": "MURA",
        "urls": {},
        "kaggle_handle": "N/A",
        "manual_source": "https://stanfordmlgroup.github.io/competitions/mura/",
        "manual_note": "Register at Stanford ML Group website to download MURA",
    },
}


def generate_download_block(config, dataset_name):
    """Generate the new download section for a script."""
    required_file = config["required_file"]
    urls = config.get("urls", {})
    kaggle_handle = config.get("kaggle_handle", "N/A")
    is_competition = config.get("is_competition", False)
    manual_source = config.get("manual_source", "")
    manual_note = config.get("manual_note", "")
    
    has_direct_urls = len(urls) > 0
    is_kaggle_only = kaggle_handle != "N/A" and not has_direct_urls
    is_manual = kaggle_handle == "N/A" and not has_direct_urls

    block = f'''# ── 1. DISCOVER OR DOWNLOAD DATA ───────────────────────────────────────────

REQUIRED_FILE = "{required_file}"

def find_file_recursive(base_path, target):
    """Walk a directory tree and return True if target file/folder exists."""
    for root, dirs, files in os.walk(base_path):
        if target in files or target in dirs:
            return True
    return False

def find_kaggle_dataset_path():
    """Search /kaggle/input for a directory that contains our required file/folder."""
    base = "/kaggle/input"
    if not os.path.exists(base):
        return None
    # Check exact slug match first
    exact = os.path.join(base, DATASET_SLUG)
    if os.path.isdir(exact) and find_file_recursive(exact, REQUIRED_FILE):
        return exact
    # Check every directory in /kaggle/input
    for d in sorted(os.listdir(base)):
        candidate = os.path.join(base, d)
        if os.path.isdir(candidate) and find_file_recursive(candidate, REQUIRED_FILE):
            return candidate
    return None

'''
    # Add download function for Category A (direct URLs)
    if has_direct_urls:
        urls_dict_str = "{\n"
        for fname, url in urls.items():
            urls_dict_str += f'        "{fname}": "{url}",\n'
        urls_dict_str += "    }"
        
        block += f'''def download_dataset(dest_path):
    """Download {dataset_name} directly from official source (no Kaggle API needed)."""
    os.makedirs(dest_path, exist_ok=True)
    URLS = {urls_dict_str}
    
    for fname, url in URLS.items():
        out_path = os.path.join(dest_path, fname)
        if os.path.exists(out_path):
            print(f"  Already exists: {{fname}}")
            continue
        print(f"  Downloading {{fname}} ...")
        ret = os.system(f\'wget -q --show-progress -O "{{out_path}}" "{{url}}"\')
        if ret != 0:
            ret = os.system(f\'curl -L -o "{{out_path}}" "{{url}}"\')
        if ret != 0:
            try:
                import urllib.request
                urllib.request.urlretrieve(url, out_path)
            except Exception as e:
                print(f"  ERROR: Download failed for {{fname}}: {{e}}")
                return False
        print(f"  Done: {{fname}}")
    
    # Unzip any zip files
    import zipfile
    for fname in list(os.listdir(dest_path)):
        if fname.endswith(".zip"):
            zp = os.path.join(dest_path, fname)
            print(f"  Extracting {{fname}}...")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(dest_path)
            os.remove(zp)
            print(f"  Extraction complete.")
    
    return find_file_recursive(dest_path, REQUIRED_FILE)

'''

    # Main discovery logic
    block += '''if not DATASET_PATH:
    # Step 1: Check if dataset is already attached in /kaggle/input
    kaggle_path = find_kaggle_dataset_path()
    
    if kaggle_path:
        DATASET_PATH = kaggle_path
        print(f"Found {DATASET_NAME} dataset at: {DATASET_PATH}")
'''

    if has_direct_urls:
        # Category A: Try direct download
        block += '''    else:
        # Step 2: Download from original source
        if os.path.exists("/kaggle/working"):
            dest = f"/kaggle/working/{DATASET_NAME}_data"
        else:
            dest = os.path.join(BASE_DIR, 'data')
        
        if find_file_recursive(dest, REQUIRED_FILE):
            DATASET_PATH = dest
            print(f"Dataset already downloaded at: {DATASET_PATH}")
        else:
            print(f"Downloading {DATASET_NAME} from official source...")
            success = download_dataset(dest)
            if success:
                DATASET_PATH = dest
                print("Download complete!")
            else:
                print("\\nERROR: Download failed. Please enable Internet in Kaggle Notebook settings.")
'''
        # Also mention Add Data as fallback if kaggle_handle exists
        if kaggle_handle != "N/A":
            block += f'''                print("Or click '+ Add Data' and search for: {kaggle_handle}")
'''
        block += '''                sys.exit(1)
'''
    elif is_kaggle_only:
        # Category B: Must attach via Add Data
        block += f'''    else:
        # This dataset is ONLY available on Kaggle. You must attach it.
        print("\\n" + "!"*70)
        print("DATASET NOT FOUND!")
        print("!"*70)
        if os.path.exists("/kaggle/input"):
            available = os.listdir("/kaggle/input")
            print(f"\\nCurrently attached in /kaggle/input: {{available}}")
        print("\\n>>> HOW TO FIX <<<")
        print("1. Click '+ Add Data' on the right panel of your Kaggle Notebook.")
        print("2. Search for: {kaggle_handle}")
        print("3. Click the '+' button to attach it.")
        print("4. Re-run this notebook.")
        print("!"*70)
        sys.exit(1)
'''
    elif is_manual:
        # Category C: Manual registration required
        block += f'''    else:
        print("\\n" + "!"*70)
        print("MANUAL DOWNLOAD REQUIRED!")
        print("!"*70)
        print("\\nThis dataset requires registration/credentials to download.")
        print("Source: {manual_source}")
        print("Note:   {manual_note}")
        print("\\nAfter downloading, upload the dataset to your Kaggle Notebook")
        print("via '+ Add Data' > 'Upload' and re-run this notebook.")
        print("!"*70)
        sys.exit(1)
'''

    block += '''
print(f"DATASET_PATH = {DATASET_PATH}")
print(f"Contents: {os.listdir(DATASET_PATH)[:20]}")

'''
    return block


def patch_script(filepath, config, dataset_name):
    """Patch a single script file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_block = generate_download_block(config, dataset_name)
    
    # Escape backslashes in replacement string so re.sub doesn't process them
    safe_new_block = new_block.replace('\\', '\\\\')
    
    # Replace the old download section
    new_content = OLD_DOWNLOAD_PATTERN.sub(safe_new_block, content, count=1)
    
    if new_content == content:
        print(f"  WARNING: Pattern not matched in {os.path.basename(filepath)}")
        return False
        
    if len(new_content) < 1000:
        print(f"  ERROR: new_content is suspiciously small! Aborting write.")
        return False
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    patched = 0
    skipped = 0
    failed = 0
    
    for script_file in sorted(os.listdir(SCRIPTS_DIR)):
        if not script_file.endswith('.py'):
            continue
        
        filepath = os.path.join(SCRIPTS_DIR, script_file)
        
        if script_file not in DOWNLOAD_CONFIGS:
            print(f"  SKIP (no config): {script_file}")
            skipped += 1
            continue
        
        config = DOWNLOAD_CONFIGS[script_file]
        if config is None:
            print(f"  SKIP (already fixed): {script_file}")
            skipped += 1
            continue
        
        dataset_name = script_file.replace("run_", "").replace(".py", "").upper()
        
        print(f"  Patching: {script_file} ...", end=" ")
        success = patch_script(filepath, config, dataset_name)
        if success:
            # Verify syntax
            import py_compile
            try:
                py_compile.compile(filepath, doraise=True)
                print("OK (syntax verified)")
                patched += 1
            except py_compile.PyCompileError as e:
                print(f"SYNTAX ERROR: {e}")
                failed += 1
        else:
            print("FAILED (pattern not matched)")
            failed += 1
    
    print(f"\nDone! Patched: {patched}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
