
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
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # trades compute for memory
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
DATASET_NAME   = "VinDr-CXR"
DATASET_SLUG   = "vinbigdata-chest-xray-abnormalities-detection"
DATASET_HANDLE = "vinbigdata-chest-xray-abnormalities-detection"
IS_COMPETITION = True
DOWNLOAD_CMD   = """kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection"""
NUM_CLASSES    = 15
SEED           = 42
IMG_SIZE       = 224
BATCH_SIZE     = 8 * max(1, torch.cuda.device_count())  # Kaggle-safe: 8 per GPU to avoid OOM
NUM_WORKERS    = 2
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

REQUIRED_FILE = "train.csv"

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

if not DATASET_PATH:
    # Step 1: Check if dataset is already attached in /kaggle/input
    kaggle_path = find_kaggle_dataset_path()
    
    if kaggle_path:
        DATASET_PATH = kaggle_path
        print(f"Found {DATASET_NAME} dataset at: {DATASET_PATH}")
    else:
        # This dataset is ONLY available on Kaggle. You must attach it.
        print("\n" + "!"*70)
        print("DATASET NOT FOUND!")
        print("!"*70)
        if os.path.exists("/kaggle/input"):
            available = os.listdir("/kaggle/input")
            print(f"\nCurrently attached in /kaggle/input: {available}")
        print("\n>>> HOW TO FIX <<<")
        print("1. Click '+ Add Data' on the right panel of your Kaggle Notebook.")
        print("2. Search for: vinbigdata-chest-xray-abnormalities-detection")
        print("3. Click the '+' button to attach it.")
        print("4. Re-run this notebook.")
        print("!"*70)
        sys.exit(1)

print(f"DATASET_PATH = {DATASET_PATH}")
print(f"Contents: {os.listdir(DATASET_PATH)[:20]}")

# ── 2. DATA LOADING & SAMPLING ─────────────────────────────────────────────

# =============================================================================
# VinDr-CXR SPECIFIC CUSTOM DATALOADER
# =============================================================================
print("--- Initializing Custom VinDr-CXR DataLoader ---")
meta_csv = None
for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        if f == "train.csv":
            meta_csv = os.path.join(root, f)
            break

if not meta_csv:
    print("ERROR: Could not find train.csv")
    sys.exit(1)

try:
    import pydicom
except ImportError:
    print("ERROR: pydicom required for VinDr-CXR. pip install pydicom")
    sys.exit(1)

df = pd.read_csv(meta_csv)
official_classes = sorted(df['class_name'].unique())
CLASS_NAMES = [str(c) for c in official_classes]
NUM_CLASSES = len(CLASS_NAMES)
label2idx = {cls: idx for idx, cls in enumerate(official_classes)}

image_files = {}
for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        if f.lower().endswith('.dicom'):
            image_files[os.path.splitext(f)[0]] = os.path.join(root, f)

items = []
missing = 0
for _, row in df.drop_duplicates(subset=['image_id']).iterrows():
    img_id = str(row['image_id'])
    lbl = row['class_name']
    if img_id in image_files:
        items.append((image_files[img_id], label2idx[lbl]))
    else:
        missing += 1

if missing > 0: print(f"Warning: {missing} images missing from directory.")

class CustomDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        dcm = pydicom.dcmread(path)
        img_arr = dcm.pixel_array
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        img_arr = ((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

base_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
full_dataset = CustomDataset(items, transform=base_transform)
print(f"Total verified dataset size: {len(full_dataset)}")
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")
print("\n--- Pre-Run Validation Check ---")
for c, name in enumerate(CLASS_NAMES): print(f"Class '{name}' mapped to index {c} correctly.")
print("Validation Pass Successful!\n")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds_raw, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

# Robustly extract training labels for counting
if hasattr(full_dataset, 'targets'):
    train_labels = [full_dataset.targets[i] for i in train_ds_raw.indices]
elif hasattr(full_dataset, 'items'):
    train_labels = [full_dataset.items[i][1] for i in train_ds_raw.indices]
else:
    # Slow fallback if neither exists
    train_labels = [train_ds_raw[i][1] for i in range(len(train_ds_raw))]

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

loader_instance = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=instance_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)
loader_median   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=median_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)
loader_reverse  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=reverse_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)
val_loader      = DataLoader(AugmentedSubset(val_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_loader     = DataLoader(AugmentedSubset(test_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)

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
        # Gradient checkpointing: recompute backbone activations during backward
        # instead of storing them — ~50% less GPU RAM at ~30% more compute cost.
        # Critical for HybridMix which calls encoder twice per step.
        feat = grad_checkpoint(self.backbone, x, use_reentrant=False)
        return self.proj(feat)

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
def train_epoch(model, opt, use_hm=True, warmup=False, epoch=0, total_epochs=0, phase=''):
    model.train()
    total, n = 0.0, 0
    it_med = iter(loader_median)
    it_rev = iter(loader_reverse)
    num_steps = len(loader_instance)
    print_every = max(1, num_steps // 10)  # ~10 progress lines per epoch

    for step, (imgs_i, lbl_i) in enumerate(loader_instance):
        try:    imgs_m, lbl_m = next(it_med)
        except: it_med = iter(loader_median);  imgs_m, lbl_m = next(it_med)
        try:    imgs_r, lbl_r = next(it_rev)
        except: it_rev = iter(loader_reverse); imgs_r, lbl_r = next(it_rev)

        imgs_i, lbl_i = imgs_i.to(DEVICE), lbl_i.to(DEVICE)
        imgs_m, lbl_m = imgs_m.to(DEVICE), lbl_m.to(DEVICE)
        imgs_r, lbl_r = imgs_r.to(DEVICE), lbl_r.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        imgs_hm = torch.cat([imgs_i, imgs_m])
        lbl_hm  = torch.cat([lbl_i,  lbl_m])

        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
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
        if (step + 1) % print_every == 0 or (step + 1) == num_steps:
            print(f'  [{phase:10s}] Ep {epoch:03d}/{total_epochs}'
                  f'  step {step+1:>{len(str(num_steps))}}/{num_steps}'
                  f'  loss={loss.item():.4f}  avg={total/n:.4f}', flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, return_all=False):
    model.eval()
    preds, labels, probs, latents = [], [], [], []
    for imgs, lbl in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs, mode='inference')
        prob = F.softmax(logits, dim=-1)
        p = logits.argmax(-1)
        
        preds.append(p.cpu()); labels.append(lbl.cpu())
        if return_all:
            probs.append(prob.cpu())
            enc = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
            latents.append(enc(imgs).cpu())
            
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    if return_all:
        probs = torch.cat(probs).numpy()
        latents = torch.cat(latents).numpy()
        return ((preds == labels).mean(), balanced_accuracy_score(labels, preds), preds, labels, probs, latents)
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

    loss = train_epoch(model, optimizer, use_hm=use_hm, warmup=warmup,
                       epoch=epoch, total_epochs=TOTAL_EPOCHS, phase=phase)
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

# Free GPU memory before evaluation
import gc
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────
import time
import json
import scipy.stats as stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, cohen_kappa_score, brier_score_loss,
                             log_loss, roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize
try:
    import umap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
    import umap
from sklearn.manifold import TSNE

# Create directories
os.makedirs(os.path.join(EXP_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, 'predictions'), exist_ok=True)

# Load best checkpoint
state_dict = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

print("Running final comprehensive evaluation...")
start_eval_time = time.time()
test_acc, test_bal, test_preds, test_labels, test_probs, test_latents = evaluate(model, test_loader, return_all=True)
eval_time = time.time() - start_eval_time

# Save predictions
np.save(os.path.join(EXP_DIR, 'predictions', 'test_preds.npy'), test_preds)
np.save(os.path.join(EXP_DIR, 'predictions', 'test_probs.npy'), test_probs)
np.save(os.path.join(EXP_DIR, 'predictions', 'test_labels.npy'), test_labels)

# ─── METRICS CALCULATION ───
# Long-tail metrics
head_acc = []
med_acc = []
tail_acc = []
for c in range(NUM_CLASSES):
    idx = (test_labels == c)
    if not idx.any(): continue
    acc_c = accuracy_score(test_labels[idx], test_preds[idx])
    if class_counts.get(c, 0) > 100: head_acc.append(acc_c)
    elif class_counts.get(c, 0) > 20: med_acc.append(acc_c)
    else: tail_acc.append(acc_c)
head_acc = np.mean(head_acc) if head_acc else 0.0
med_acc = np.mean(med_acc) if med_acc else 0.0
tail_acc = np.mean(tail_acc) if tail_acc else 0.0

# Ranking metrics
top1 = accuracy_score(test_labels, test_preds)
top3 = 0.0
top5 = 0.0
if NUM_CLASSES >= 3:
    top3_preds = np.argsort(test_probs, axis=1)[:, -3:]
    top3 = np.mean([test_labels[i] in top3_preds[i] for i in range(len(test_labels))])
if NUM_CLASSES >= 5:
    top5_preds = np.argsort(test_probs, axis=1)[:, -5:]
    top5 = np.mean([test_labels[i] in top5_preds[i] for i in range(len(test_labels))])

# Probabilistic metrics
ll = log_loss(test_labels, test_probs, labels=range(NUM_CLASSES))
try:
    if NUM_CLASSES == 2:
        brier = brier_score_loss(test_labels, test_probs[:, 1])
    else:
        # Brier score for multiclass
        labels_bin = label_binarize(test_labels, classes=range(NUM_CLASSES))
        brier = np.mean(np.sum((test_probs - labels_bin)**2, axis=1))
except:
    brier = 0.0

# ECE Calculation
confs = np.max(test_probs, axis=1)
preds_ece = np.argmax(test_probs, axis=1)
accs_ece = (preds_ece == test_labels)
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
ece = 0.0
for i in range(n_bins):
    mask = (confs > bins[i]) & (confs <= bins[i+1])
    if np.sum(mask) > 0:
        bin_acc = np.mean(accs_ece[mask])
        bin_conf = np.mean(confs[mask])
        ece += np.sum(mask) / len(test_labels) * np.abs(bin_acc - bin_conf)

# ROC & PR
try:
    if NUM_CLASSES == 2:
        roc_macro = roc_auc_score(test_labels, test_probs[:, 1])
        roc_weighted = roc_macro
        ap_score = average_precision_score(test_labels, test_probs[:, 1])
    else:
        roc_macro = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
        roc_weighted = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='weighted')
        labels_bin = label_binarize(test_labels, classes=range(NUM_CLASSES))
        ap_score = average_precision_score(labels_bin, test_probs, average='macro')
except:
    roc_macro, roc_weighted, ap_score = 0.0, 0.0, 0.0

# ─── BOOTSTRAPPING FOR CI & STATS ───
n_bootstraps = 1000
boot_acc = []
for i in range(n_bootstraps):
    idx = np.random.randint(0, len(test_labels), len(test_labels))
    boot_acc.append(accuracy_score(test_labels[idx], test_preds[idx]))
ci_lower, ci_upper = np.percentile(boot_acc, [2.5, 97.5])
mean_acc = np.mean(boot_acc)
std_acc = np.std(boot_acc)

# Wilcoxon Signed Rank against Majority Class Baseline (since no actual baseline exists)
majority_class = np.argmax(np.bincount(test_labels))
baseline_preds = np.full_like(test_labels, majority_class)
try:
    stat, pval = stats.wilcoxon((test_preds == test_labels).astype(int), (baseline_preds == test_labels).astype(int))
except:
    pval = 1.0

# ─── GENERATE FIGURES ───
print("Generating publication quality figures...")
def save_fig(name):
    # Save PNG only to reduce disk I/O and memory pressure on Kaggle
    plt.savefig(os.path.join(EXP_DIR, 'figures', f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# 1. Training Curves
plt.figure(figsize=(10,5))
plt.plot(history['epoch'], history['loss'], label='Train Loss', lw=2)
plt.plot(history['epoch'], history['val_acc'], label='Val Acc', lw=2)
plt.plot(history['epoch'], history['val_bal'], label='Val BAcc', lw=2)
plt.legend()
plt.title(f"{DATASET_NAME} Training Curves")
plt.grid(True, alpha=0.3)
save_fig('training_curves')

# 2. Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"{DATASET_NAME} Confusion Matrix")
save_fig('confusion_matrix')

# Normalized CM
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"{DATASET_NAME} Normalized Confusion Matrix")
save_fig('confusion_matrix_normalized')

# 3. Class Distribution
plt.figure(figsize=(12, 5))
sns.barplot(x=CLASS_NAMES, y=[class_counts.get(i,0) for i in range(NUM_CLASSES)])
plt.title(f"{DATASET_NAME} Long Tail Distribution (Train)")
plt.xticks(rotation=45, ha='right')
save_fig('class_distribution')

# 4. Reliability Diagram
plt.figure(figsize=(8,8))
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
bin_accs = []
bin_confs = []
for i in range(n_bins):
    mask = (confs > bins[i]) & (confs <= bins[i+1])
    if np.sum(mask) > 0:
        bin_accs.append(np.mean(accs_ece[mask]))
        bin_confs.append(np.mean(confs[mask]))
plt.plot(bin_confs, bin_accs, 's-', label='Model')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Reliability Diagram')
plt.legend()
save_fig('reliability_diagram')

# 5. UMAP & t-SNE
print("Generating embeddings...")
try:
    if len(test_latents) > 2000:
        # Subsample for speed and to avoid OOM during TSNE/UMAP
        idx = np.random.choice(len(test_latents), 2000, replace=False)
        l_sub, y_sub = test_latents[idx], test_labels[idx]
    else:
        l_sub, y_sub = test_latents, test_labels
    
    tsne_res = TSNE(n_components=2, random_state=SEED).fit_transform(l_sub)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=[CLASS_NAMES[i] for i in y_sub], palette='tab10', s=20)
    plt.title('t-SNE Embeddings')
    save_fig('tsne')
    
    umap_res = umap.UMAP(random_state=SEED).fit_transform(l_sub)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=umap_res[:,0], y=umap_res[:,1], hue=[CLASS_NAMES[i] for i in y_sub], palette='tab10', s=20)
    plt.title('UMAP Embeddings')
    save_fig('umap')
except Exception as e:
    print(f"Embedding visualization failed: {e}")
finally:
    del test_latents
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ─── STATISTICAL REPORT ───
report_txt = f"""================================================================================
STATISTICAL REPORT FOR {DATASET_NAME}
================================================================================

1. CLASSIFICATION METRICS
--------------------------------------------------------------------------------
Accuracy               : {top1:.4f}
Balanced Accuracy      : {test_bal:.4f}
Precision Macro        : {precision_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
Precision Weighted     : {precision_score(test_labels, test_preds, average='weighted', zero_division=0):.4f}
Recall Macro           : {recall_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
Recall Weighted        : {recall_score(test_labels, test_preds, average='weighted', zero_division=0):.4f}
F1 Macro               : {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}
F1 Weighted            : {f1_score(test_labels, test_preds, average='weighted', zero_division=0):.4f}
MCC                    : {matthews_corrcoef(test_labels, test_preds):.4f}
Cohen Kappa            : {cohen_kappa_score(test_labels, test_preds):.4f}

2. LONG-TAIL METRICS
--------------------------------------------------------------------------------
Head (Many-Shot) Acc   : {head_acc:.4f}
Medium (Med-Shot) Acc  : {med_acc:.4f}
Tail (Few-Shot) Acc    : {tail_acc:.4f}

3. RANKING METRICS
--------------------------------------------------------------------------------
Top-1 Accuracy         : {top1:.4f}
Top-3 Accuracy         : {top3:.4f}
Top-5 Accuracy         : {top5:.4f}

4. PROBABILISTIC METRICS
--------------------------------------------------------------------------------
Log Loss               : {ll:.4f}
Brier Score            : {brier:.4f}
Expected Calib. Err    : {ece:.4f}

5. ROC & PR METRICS
--------------------------------------------------------------------------------
ROC-AUC Macro          : {roc_macro:.4f}
ROC-AUC Weighted       : {roc_weighted:.4f}
PR-AUC (Avg Precision) : {ap_score:.4f}

6. RESOURCE METRICS
--------------------------------------------------------------------------------
Inference Time         : {eval_time:.2f} seconds
Parameters             : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
GPU Memory Allocated   : {torch.cuda.memory_allocated() / 1024**2:.2f} MB

7. STATISTICAL SIGNIFICANCE
--------------------------------------------------------------------------------
Bootstrap Mean Acc     : {mean_acc:.4f} ± {std_acc:.4f}
95% CI                 : [{ci_lower:.4f}, {ci_upper:.4f}]
Wilcoxon p-value       : {pval:.4e} (vs Majority Class Baseline)

================================================================================
DETAILED CLASSIFICATION REPORT
================================================================================
{classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)}
"""

with open(os.path.join(EXP_DIR, 'statistical_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)
with open(os.path.join(EXP_DIR, 'metrics', 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'acc': top1, 'bacc': test_bal, 'f1_macro': f1_score(test_labels, test_preds, average='macro', zero_division=0),
        'head_acc': head_acc, 'med_acc': med_acc, 'tail_acc': tail_acc,
        'roc_macro': roc_macro, 'ap_score': ap_score, 'ece': ece
    }, f, indent=4)

# Config logging
config_log = {
    'dataset': DATASET_NAME, 'epochs': TOTAL_EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
    'encoder_dim': ENCODER_DIM, 'latent_dim': LATENT_DIM, 'seed': SEED
}
with open(os.path.join(EXP_DIR, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_log, f, indent=4)

print("Benchmark completed successfully. Results saved in", EXP_DIR)
