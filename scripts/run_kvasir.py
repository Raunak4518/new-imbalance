
import os, random, warnings, sys, subprocess, shutil
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score, roc_auc_score,
                              f1_score, precision_score, recall_score,
                              cohen_kappa_score, matthews_corrcoef, brier_score_loss, log_loss)

# ── USER CONFIGURATION ─────────────────────────────────────────────────────
# Modify this path to point to your local dataset folder.
# If left empty, the script will attempt to detect Kaggle or download automatically.
DATASET_PATH   = ""

# ── CONFIGURATION & HYPERPARAMETERS ────────────────────────────────────────
DATASET_NAME   = "Kvasir"
DATASET_SLUG   = "kvasir-dataset"
DATASET_HANDLE = "meetnagadia/kvasir-dataset"
IS_COMPETITION = False
DOWNLOAD_CMD   = """kaggle datasets download meetnagadia/kvasir-dataset"""
NUM_CLASSES    = 8
SEED           = 42
IMG_SIZE       = 224
BATCH_SIZE     = 16 * max(1, torch.cuda.device_count()) # Scale batch size with GPUs
NUM_WORKERS    = 4
ENCODER_DIM    = 512
LATENT_DIM     = 128
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
EPOCHS_WARMUP  = 15
EPOCHS_PHASE1  = 20
EPOCHS_PHASE2  = 10
TEMPERATURE    = 0.07
CVAE_BETA      = 0.5
GRAD_CLIP      = 2.0
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR       = os.path.abspath(f'./results_{DATASET_NAME}')
EXP_DIR        = os.path.join(BASE_DIR, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

print(f"--- Running Benchmark for {DATASET_NAME} ---")
print(f"Device: {DEVICE}")
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")

# ── 1. DISCOVER OR DOWNLOAD DATA ───────────────────────────────────────────

REQUIRED_FILE = "dyed-lifted-polyps"

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

def download_dataset(dest_path):
    """Download KVASIR directly from official source (no Kaggle API needed)."""
    os.makedirs(dest_path, exist_ok=True)
    URLS = {
        "kvasir-dataset-v2.zip": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
    }
    
    for fname, url in URLS.items():
        out_path = os.path.join(dest_path, fname)
        if os.path.exists(out_path):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname} ...")
        ret = os.system(f'wget -q --show-progress -O "{out_path}" "{url}"')
        if ret != 0:
            ret = os.system(f'curl -L -o "{out_path}" "{url}"')
        if ret != 0:
            try:
                import urllib.request
                urllib.request.urlretrieve(url, out_path)
            except Exception as e:
                print(f"  ERROR: Download failed for {fname}: {e}")
                return False
        print(f"  Done: {fname}")
    
    # Unzip any zip files
    import zipfile
    for fname in list(os.listdir(dest_path)):
        if fname.endswith(".zip"):
            zp = os.path.join(dest_path, fname)
            print(f"  Extracting {fname}...")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(dest_path)
            os.remove(zp)
            print(f"  Extraction complete.")
    
    return find_file_recursive(dest_path, REQUIRED_FILE)

if not DATASET_PATH:
    # Step 1: Check if dataset is already attached in /kaggle/input
    kaggle_path = find_kaggle_dataset_path()
    
    if kaggle_path:
        DATASET_PATH = kaggle_path
        print(f"Found {DATASET_NAME} dataset at: {DATASET_PATH}")
    else:
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
                print("\nERROR: Download failed. Please enable Internet in Kaggle Notebook settings.")
                print("Or click '+ Add Data' and search for: meetnagadia/kvasir-dataset")
                sys.exit(1)

print(f"DATASET_PATH = {DATASET_PATH}")
print(f"Contents: {os.listdir(DATASET_PATH)[:20]}")

# ── 2. DATA LOADING & SAMPLING ─────────────────────────────────────────────
# Find the directory containing the actual images
img_root = DATASET_PATH
image_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if len(imgs) > len(image_files):
        image_files = [os.path.join(root, f) for f in imgs]
        img_root = root

class_dirs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class GenericDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

if len(class_dirs) >= 2:
    print(f"Found {len(class_dirs)} class subdirectories. Using ImageFolder.")
    full_dataset = datasets.ImageFolder(root=img_root, transform=base_transform)
    CLASS_NAMES = full_dataset.classes
    NUM_CLASSES = len(CLASS_NAMES)
else:
    print("No class subdirectories found. Searching for metadata CSV...")
    csv_files = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.endswith('.csv'): csv_files.append(os.path.join(root, f))
    
    if not csv_files:
        print(f"ERROR: No class subdirectories and no CSV file found in {DATASET_PATH}. Cannot load dataset.")
        sys.exit(1)
    
    best_csv = csv_files[0]
    for c in csv_files:
        if 'train' in c.lower() or 'meta' in c.lower() or 'groundtruth' in c.lower() or 'labels' in c.lower():
            best_csv = c
            break
            
    print(f"Using CSV: {best_csv}")
    df = pd.read_csv(best_csv)
    
    img_col = None
    for col in df.columns:
        if 'image' in col.lower() or 'id' in col.lower() or 'name' in col.lower():
            img_col = col; break
    if not img_col: img_col = df.columns[0]
    
    label_col = None
    for col in df.columns:
        if col != img_col and ('class' in col.lower() or 'label' in col.lower() or 'dx' in col.lower() or 'target' in col.lower() or 'diagnosis' in col.lower()):
            label_col = col; break
    
    if not label_col:
        potential_classes = [c for c in df.columns if c != img_col and pd.api.types.is_numeric_dtype(df[c])]
        if potential_classes:
            print(f"Assuming one-hot encoded labels across columns: {potential_classes}")
            df['guessed_label'] = df[potential_classes].idxmax(axis=1)
            label_col = 'guessed_label'
        else:
            label_col = df.columns[-1]

    print(f"Guessed Image Column: '{img_col}', Label Column: '{label_col}'")
    
    unique_labels = sorted(df[label_col].dropna().unique())
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    CLASS_NAMES = [str(l) for l in unique_labels]
    NUM_CLASSES = len(CLASS_NAMES)
    
    items = []
    filename_map = {os.path.basename(p): p for p in image_files}
    filename_map_noext = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    
    for _, row in df.iterrows():
        img_id = str(row[img_col])
        lbl = row[label_col]
        if pd.isna(lbl): continue
        
        path = None
        if img_id in filename_map: path = filename_map[img_id]
        elif img_id in filename_map_noext: path = filename_map_noext[img_id]
        else:
            for ext in ['.jpg', '.jpeg', '.png']:
                if img_id+ext in filename_map:
                    path = filename_map[img_id+ext]
                    break
                    
        if path: items.append((path, label2idx[lbl]))
            
    if not items:
        print("ERROR: Could not match any images from the CSV to the actual image files.")
        sys.exit(1)

    print(f"Successfully matched {len(items)} images from CSV.")
    full_dataset = GenericDataset(items, transform=base_transform)

print(f"Total dataset size: {len(full_dataset)}")
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds_raw, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

train_labels = [full_dataset.targets[i] for i in train_ds_raw.indices]
class_counts = Counter(train_labels)

sorted_counts = sorted(class_counts.values(), reverse=True)
HEAD_THRESH = sorted_counts[len(sorted_counts)//4] if len(sorted_counts)>4 else sorted_counts[0]
TAIL_THRESH = sorted_counts[-max(1, len(sorted_counts)//4)] if len(sorted_counts)>4 else sorted_counts[-1]

CLASS_GROUPS = {CLASS_NAMES[c]:('head' if class_counts[c] > HEAD_THRESH else 'tail' if class_counts[c] < TAIL_THRESH else 'median') for c in range(NUM_CLASSES)}
COLORS = {'head':'#2ecc71','median':'#f39c12','tail':'#e74c3c'}

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_transform = transforms.Compose([
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class AugmentedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = AugmentedSubset(train_ds_raw, transform=train_transform)

def instance_sampler(labels): return WeightedRandomSampler(torch.ones(len(labels)), len(labels), replacement=True)
def median_sampler(labels):
    counts = Counter(labels)
    median_n = sorted(counts.values())[len(counts)//2]
    w = np.clip(median_n / np.array([counts[l] for l in labels], dtype=float), 1.0, 5.0)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float), len(labels), replacement=True)
def reverse_sampler(labels):
    counts = Counter(labels)
    w = np.clip(1.0 / np.array([counts[l] for l in labels], dtype=float), 0, 8)
    w = w / w.sum() * len(w)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float), len(labels), replacement=True)

loader_instance = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=instance_sampler(train_labels), num_workers=NUM_WORKERS)
loader_median   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=median_sampler(train_labels), num_workers=NUM_WORKERS)
loader_reverse  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=reverse_sampler(train_labels), num_workers=NUM_WORKERS)
val_loader      = DataLoader(AugmentedSubset(val_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader     = DataLoader(AugmentedSubset(test_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ── 3. EXACT ARCHITECTURE (STTPNet) ────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, out_dim=ENCODER_DIM):
        super().__init__()
        bb    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        bb.fc = nn.Identity()
        self.backbone = bb
        self.proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim))

    def forward(self, x):
        return self.proj(self.backbone(x))

class CVAE(nn.Module):
    def __init__(self, feat_dim=ENCODER_DIM, lat_dim=LATENT_DIM, nc=NUM_CLASSES):
        super().__init__()
        self.lat_dim = lat_dim
        self.nc      = nc
        self.enc1    = nn.Linear(feat_dim + nc, 256)
        self.mu_fc   = nn.Linear(256, lat_dim)
        self.lv_fc   = nn.Linear(256, lat_dim)
        self.dec1    = nn.Linear(lat_dim + nc, 256)
        self.dec_out = nn.Linear(256, feat_dim)

    def encode(self, f, y_oh):
        h = F.relu(self.enc1(torch.cat([f, y_oh], dim=-1)))
        return self.mu_fc(h), self.lv_fc(h)

    def decode(self, z, y_oh):
        return self.dec_out(F.relu(self.dec1(torch.cat([z, y_oh], dim=-1))))

    def forward(self, f, labels):
        y_oh      = F.one_hot(labels, self.nc).float().to(f.device)
        mu, lv    = self.encode(f, y_oh)
        z         = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        return self.decode(z, y_oh), mu, lv, z

def cvae_loss(recon, orig, mu, lv, beta=CVAE_BETA):
    return (F.mse_loss(recon, orig) +
            beta * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())))

class SupConLoss(nn.Module):
    def __init__(self, temp=TEMPERATURE):
        super().__init__()
        self.temp = temp

    def forward(self, features, labels):
        f    = F.normalize(features, dim=1)
        B    = f.size(0)
        mask = torch.eq(labels.unsqueeze(0),
                        labels.unsqueeze(1)).float().to(f.device)
        mask.fill_diagonal_(0)
        sim      = torch.mm(f, f.T) / self.temp
        sim      = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim  = torch.exp(sim)
        log_prob = sim - torch.log(
            exp_sim.sum(1, keepdim=True) - exp_sim.diagonal().unsqueeze(1) + 1e-8)
        n_pos = mask.sum(1).clamp(min=1)
        return -(mask * log_prob).sum(1).div(n_pos).mean()

def pixel_cutmix(imgs, labels, alpha=1.0):
    B, C, H, W = imgs.shape
    perm     = torch.randperm(B, device=imgs.device)
    imgs_b   = imgs[perm]
    labels_b = labels[perm]
    lam      = float(np.random.beta(alpha, alpha))
    cut_rat  = np.sqrt(1.0 - lam)
    cut_w    = int(W * cut_rat)
    cut_h    = int(H * cut_rat)
    cx       = np.random.randint(W)
    cy       = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    imgs_mixed = imgs.clone()
    imgs_mixed[:, :, y1:y2, x1:x2] = imgs_b[:, :, y1:y2, x1:x2]
    lam_adj  = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    lam_vec  = torch.full((B,), lam_adj, device=imgs.device)
    return imgs_mixed, labels, labels_b, lam_vec

def latent_mixup(z, labels, alpha=0.4):
    B        = z.size(0)
    perm     = torch.randperm(B, device=z.device)
    z_b      = z[perm]
    labels_b = labels[perm]
    lam      = float(np.random.beta(alpha, alpha))
    z_mixed  = lam * z + (1 - lam) * z_b
    lam_vec  = torch.full((B,), lam, device=z.device)
    return z_mixed, labels, labels_b, lam_vec

class Head(nn.Module):
    def __init__(self, in_d=ENCODER_DIM, nc=NUM_CLASSES, drop=0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_d, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
            nn.Linear(256, nc))
    def forward(self, x): return self.fc(x)

class STTPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder(ENCODER_DIM)
        self.cvae     = CVAE(ENCODER_DIM, LATENT_DIM, NUM_CLASSES)
        self.head1    = Head(nc=NUM_CLASSES)
        self.head2    = Head(nc=NUM_CLASSES)
        self.sc_proj  = nn.Sequential(
            nn.Linear(ENCODER_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128))
        self.pre_clf  = nn.Linear(ENCODER_DIM, NUM_CLASSES)

    def forward(self, imgs, labels=None, mode='inference', use_hm=True):
        """Unified forward pass to support nn.DataParallel multi-GPU routing"""
        if mode == 'head_median':
            return self.forward_head_median(imgs, labels, use_hm)
        elif mode == 'tail':
            return self.forward_tail(imgs, labels, use_hm)
        elif mode == 'warmup':
            f_hm = self.encoder(imgs)
            pre_logits = self.pre_clf(f_hm)
            l1 = self.head1(f_hm)
            # labels is passed as (imgs_hm, imgs_r) during warmup hack
            l2_w = self.head2(self.encoder(labels)) 
            return f_hm, pre_logits, l1, l2_w
        else:
            return self.fused_logits(imgs)

    def forward_head_median(self, imgs, labels, use_hm=True):
        la, lb = labels, labels
        lam_vec = torch.ones(len(labels), device=imgs.device)
        if use_hm: imgs_mix, la, lb, lam_vec = pixel_cutmix(imgs, labels)
        else: imgs_mix = imgs
        f = self.encoder(imgs_mix)
        pre_logits = self.pre_clf(f)
        sc_feat = self.sc_proj(f)
        if use_hm:
            lam_e = lam_vec.unsqueeze(1)
            y_oh_la = F.one_hot(la, NUM_CLASSES).float().to(imgs.device)
            y_oh_lb = F.one_hot(lb, NUM_CLASSES).float().to(imgs.device)
            y_oh_mix = lam_e * y_oh_la + (1 - lam_e) * y_oh_lb
        else:
            y_oh_mix = F.one_hot(la, NUM_CLASSES).float().to(imgs.device)
        mu, lv = self.cvae.encode(f, y_oh_mix)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        f_decode = self.cvae.decode(z, y_oh_mix)
        f_clean = self.encoder(imgs)
        y_oh_clean = F.one_hot(labels, NUM_CLASSES).float().to(imgs.device)
        mu_c, lv_c = self.cvae.encode(f_clean, y_oh_clean)
        z_c = mu_c + torch.randn_like(mu_c) * (0.5 * lv_c).exp()
        f_recon = self.cvae.decode(z_c, y_oh_clean)
        l1 = self.head1(f_decode)
        return l1, la, lb, lam_vec, sc_feat, mu_c, lv_c, pre_logits, f_recon, f_clean

    def forward_tail(self, imgs, labels, use_hm=True):
        f = self.encoder(imgs)
        pre_logits = self.pre_clf(f)
        y_oh = F.one_hot(labels, NUM_CLASSES).float().to(imgs.device)
        mu, lv = self.cvae.encode(f, y_oh)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        f_recon = self.cvae.decode(z, y_oh)
        la, lb = labels, labels
        lam_vec = torch.ones(len(labels), device=imgs.device)
        if use_hm:
            z_mix, la, lb, lam_vec = latent_mixup(z, labels)
            lam_e = lam_vec.unsqueeze(1)
            y_oh_la = F.one_hot(la, NUM_CLASSES).float().to(imgs.device)
            y_oh_lb = F.one_hot(lb, NUM_CLASSES).float().to(imgs.device)
            y_oh_mix = lam_e * y_oh_la + (1 - lam_e) * y_oh_lb
            f_decode = self.cvae.decode(z_mix, y_oh_mix)
        else:
            f_decode = f_recon
        l2 = self.head2(f_decode)
        return l2, la, lb, lam_vec, mu, lv, pre_logits, f_recon, f

    @torch.no_grad()
    def fused_logits(self, imgs):
        f = self.encoder(imgs)
        y_pred = self.pre_clf(f).argmax(1)
        y_oh = F.one_hot(y_pred, NUM_CLASSES).float().to(imgs.device)
        mu, _ = self.cvae.encode(f, y_oh)
        f_recon = self.cvae.decode(mu, y_oh)
        l1 = self.head1(f_recon)
        l2 = self.head2(f_recon)
        w1 = self.head1.fc[-1].weight.norm()
        w2 = self.head2.fc[-1].weight.norm()
        return (l1 / w1 + l2 / w2) * 0.5

class EBSLoss(nn.Module):
    def __init__(self, counts):
        super().__init__()
        n = torch.tensor([float(counts.get(c, 1)) for c in range(NUM_CLASSES)])
        self.register_buffer('log_n', torch.log(n))

    def forward(self, logits, labels):
        return F.cross_entropy(logits + self.log_n.to(logits.device), labels)

def mixed_ce(logits, la, lb, lam, ebs_fn):
    adj    = logits + ebs_fn.log_n.to(logits.device)
    lam    = lam.to(logits.device)
    loss_a = F.cross_entropy(adj, la.to(logits.device), reduction='none')
    loss_b = F.cross_entropy(adj, lb.to(logits.device), reduction='none')
    return (lam * loss_a + (1 - lam) * loss_b).mean()

model = STTPNet()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(DEVICE)

ebs_fn  = EBSLoss(class_counts).to(DEVICE)
sup_con = SupConLoss().to(DEVICE)

# ── 4. TRAINING LOOP ───────────────────────────────────────────────────────
def train_epoch(model, opt, use_hm=True, warmup=False):
    model.train()
    total, n = 0.0, 0
    it_med = iter(loader_median)
    it_rev = iter(loader_reverse)

    for step, (imgs_i, lbl_i) in enumerate(loader_instance):
        try:    imgs_m, lbl_m = next(it_med)
        except: it_med = iter(loader_median);  imgs_m, lbl_m = next(it_med)
        try:    imgs_r, lbl_r = next(it_rev)
        except: it_rev = iter(loader_reverse); imgs_r, lbl_r = next(it_rev)

        imgs_i, lbl_i = imgs_i.to(DEVICE), lbl_i.to(DEVICE)
        imgs_m, lbl_m = imgs_m.to(DEVICE), lbl_m.to(DEVICE)
        imgs_r, lbl_r = imgs_r.to(DEVICE), lbl_r.to(DEVICE)

        opt.zero_grad()
        imgs_hm = torch.cat([imgs_i, imgs_m])
        lbl_hm  = torch.cat([lbl_i,  lbl_m])

        if warmup:
            # We use the mode 'warmup' to route through DataParallel
            # We pass imgs_r in the labels argument slot to route it through DataParallel together
            _, pre_logits, l1, l2_w = model(imgs_hm, labels=imgs_r, mode='warmup')
            loss_pre = F.cross_entropy(pre_logits, lbl_hm)
            loss_h1 = ebs_fn(l1, lbl_hm)
            loss_h2 = ebs_fn(l2_w, lbl_r)
            loss = loss_h1 + loss_h2 + 0.5 * loss_pre
        else:
            (l1, la_hm, lb_hm, lam_hm, sc_hm, mu_hm, lv_hm, pre_hm, f_recon_hm, f_hm) = model(imgs_hm, labels=lbl_hm, mode='head_median', use_hm=use_hm)
            (l2, la_r, lb_r, lam_r, mu_r, lv_r, pre_r, f_recon_r, f_r) = model(imgs_r, labels=lbl_r, mode='tail', use_hm=use_hm)

            if use_hm:
                loss_h1 = mixed_ce(l1, la_hm, lb_hm, lam_hm, ebs_fn)
                loss_h2 = mixed_ce(l2, la_r,  lb_r,  lam_r,  ebs_fn)
            else:
                loss_h1 = ebs_fn(l1, lbl_hm)
                loss_h2 = ebs_fn(l2, lbl_r)

            head_mask = torch.tensor([lbl.item() in [c for c,g in CLASS_GROUPS.items() if g=='head'] for lbl in lbl_hm]).to(DEVICE)
            loss_head = ebs_fn(l1[head_mask], lbl_hm[head_mask]) * 0.3 if head_mask.any() else torch.tensor(0.0, device=DEVICE)

            loss_cvae = (cvae_loss(f_recon_hm, f_hm.detach(), mu_hm, lv_hm) + cvae_loss(f_recon_r, f_r.detach(), mu_r, lv_r)) * 0.5
            loss_sc = sup_con(sc_hm[:len(lbl_i)], lbl_i)
            loss_pre = (F.cross_entropy(pre_hm, lbl_hm) + F.cross_entropy(pre_r,  lbl_r)) * 0.5
            loss = loss_h1 + loss_h2 + 0.3 * loss_sc + 0.2 * loss_cvae + 0.3 * loss_pre + loss_head

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for imgs, lbl in loader:
        p = model(imgs.to(DEVICE), mode='inference').argmax(-1).cpu()
        preds.append(p); labels.append(lbl)
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return ((preds == labels).mean(), balanced_accuracy_score(labels, preds), preds, labels)

# Run Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
TOTAL_EPOCHS = EPOCHS_WARMUP + EPOCHS_PHASE1 + EPOCHS_PHASE2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

best_bal = 0.0
best_ckpt = os.path.join(EXP_DIR, 'best_model.pt')
history = {'epoch':[], 'loss':[], 'val_acc':[], 'val_bal':[], 'phase':[]}

for epoch in range(1, TOTAL_EPOCHS + 1):
    if   epoch <= EPOCHS_WARMUP: phase, warmup, use_hm = 'Warmup',     True,  False
    elif epoch <= EPOCHS_WARMUP + EPOCHS_PHASE1: phase, warmup, use_hm = 'HybridMix',  False, True
    else: phase, warmup, use_hm = 'Refinement', False, False

    loss = train_epoch(model, optimizer, use_hm=use_hm, warmup=warmup)
    acc, bal, _, _ = evaluate(model, val_loader)
    scheduler.step()

    history['epoch'].append(epoch)
    history['loss'].append(loss)
    history['val_acc'].append(acc)
    history['val_bal'].append(bal)
    history['phase'].append(phase)
    print(f'[{phase:10s}] Ep {epoch:03d}/{TOTAL_EPOCHS} loss={loss:.4f} val_acc={acc:.4f} val_bacc={bal:.4f}', flush=True)

    if bal > best_bal:
        best_bal = bal
        # Handle DataParallel state dict save correctly
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, best_ckpt)

# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────
state_dict = torch.load(best_ckpt, map_location=DEVICE)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

test_acc, test_bal, test_preds, test_labels = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}, Test Balanced Accuracy: {test_bal:.4f}")

# Generate Figures
plt.figure(figsize=(10,5))
plt.plot(history['epoch'], history['loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
plt.plot(history['epoch'], history['val_bal'], label='Val BAcc')
plt.legend()
plt.title(f"{DATASET_NAME} Training Curves")
plt.savefig(os.path.join(EXP_DIR, 'training_curves.png'), dpi=300)
plt.savefig(os.path.join(EXP_DIR, 'training_curves.svg'), format='svg')
plt.close()

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"{DATASET_NAME} Confusion Matrix")
plt.savefig(os.path.join(EXP_DIR, 'confusion_matrix.png'), dpi=300)
plt.savefig(os.path.join(EXP_DIR, 'confusion_matrix.pdf'), format='pdf')
plt.close()

# Generate Statistical Report
report_txt = f"""Statistical Report for {DATASET_NAME}
----------------------------------------
Test Accuracy          : {test_acc:.4f}
Test Balanced Accuracy : {test_bal:.4f}
Precision Macro        : {precision_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
Recall Macro           : {recall_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
F1 Macro               : {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
Cohen Kappa            : {cohen_kappa_score(test_labels, test_preds):.4f}
MCC                    : {matthews_corrcoef(test_labels, test_preds):.4f}

Detailed Classification Report:
{classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)}
"""
with open(os.path.join(EXP_DIR, 'statistical_report.txt'), 'w') as f:
    f.write(report_txt)

print("Benchmark completed successfully. Results saved in", EXP_DIR)
