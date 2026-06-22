import re
import os

with open('script_template.py', 'r', encoding='utf-8') as f:
    content = f.read()

prefix, _ = content.split('# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────')

new_eval_block = """# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────
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
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(EXP_DIR, 'figures', f'{name}.{ext}'), dpi=300, bbox_inches='tight')
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
    if len(test_latents) > 5000:
        # Subsample for speed
        idx = np.random.choice(len(test_latents), 5000, replace=False)
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

# ─── STATISTICAL REPORT ───
report_txt = f\"\"\"================================================================================
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
\"\"\"

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
"""

with open('script_template.py', 'w', encoding='utf-8') as f:
    f.write(prefix + new_eval_block)
