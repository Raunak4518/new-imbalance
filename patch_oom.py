"""
patch_oom.py  –  Apply Kaggle OOM-prevention fixes to all scripts and the template.

Changes applied to every file:
  1. BATCH_SIZE: 16 → 8 per GPU
  2. NUM_WORKERS: 4 → 2
  3. DataLoader: add pin_memory=True, prefetch_factor=2, persistent_workers=True
  4. train_epoch: add torch.amp.autocast + empty_cache at end
  5. evaluate(return_all=True): remove broken model.bottleneck call;
     collect encoder features directly (no second inference pass needed – use
     the already-computed latents from fused_logits path via a cheap re-encode)
  6. After training loop: add gc.collect() + torch.cuda.empty_cache()
  7. save_fig: only save PNG (skip PDF/SVG)
  8. t-SNE/UMAP subsample cap: 5000 → 2000
  9. Free test_latents after embedding plots
"""

import os
import re

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
TEMPLATE    = os.path.join(os.path.dirname(__file__), "script_template.py")

def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    original = src

    # ── 1. Batch size: 16 → 8 ────────────────────────────────────────────────
    src = src.replace(
        "BATCH_SIZE     = 16 * max(1, torch.cuda.device_count()) # Scale batch size with GPUs",
        "BATCH_SIZE     = 8 * max(1, torch.cuda.device_count())  # Kaggle-safe: 8 per GPU to avoid OOM"
    )

    # ── 2. NUM_WORKERS: 4 → 2 ────────────────────────────────────────────────
    src = src.replace(
        "NUM_WORKERS    = 4",
        "NUM_WORKERS    = 2"
    )

    # ── 3. DataLoader: add pin_memory + prefetch_factor + persistent_workers ──
    # We match all DataLoader(...) calls and add the kwargs if not already present.
    # Strategy: replace the closing paren pattern for each known DataLoader line.

    def add_dataloader_kwargs(text: str) -> str:
        """Add memory-efficient kwargs to DataLoader calls that are missing them."""
        # Pattern: DataLoader(... num_workers=NUM_WORKERS) possibly with shuffle=... before )
        # We target lines ending in `num_workers=NUM_WORKERS)` and add our kwargs before `)`
        pattern = r'(DataLoader\([^)]+num_workers=NUM_WORKERS)(\))'
        replacement = r'\1, pin_memory=True, prefetch_factor=2, persistent_workers=True\2'
        # Only add if not already patched
        if 'pin_memory=True' not in text:
            text = re.sub(pattern, replacement, text)
        return text

    src = add_dataloader_kwargs(src)

    # ── 4. train_epoch: AMP autocast + empty_cache at end ────────────────────
    # Wrap the forward pass section with autocast.
    # We target the opt.zero_grad() / forward / backward pattern and add autocast.

    OLD_ZERO_GRAD = "        opt.zero_grad()\n        imgs_hm = torch.cat([imgs_i, imgs_m])"
    NEW_ZERO_GRAD = "        opt.zero_grad(set_to_none=True)\n        imgs_hm = torch.cat([imgs_i, imgs_m])"
    src = src.replace(OLD_ZERO_GRAD, NEW_ZERO_GRAD)

    # Wrap warmup forward
    OLD_WARMUP_FWD = (
        "        if warmup:\n"
        "            # We use the mode 'warmup' to route through DataParallel\n"
        "            # We pass imgs_r in the labels argument slot to route it through DataParallel together\n"
        "            _, pre_logits, l1, l2_w = model(imgs_hm, labels=imgs_r, mode='warmup')\n"
        "            loss_pre = F.cross_entropy(pre_logits, lbl_hm)\n"
        "            loss_h1 = ebs_fn(l1, lbl_hm)\n"
        "            loss_h2 = ebs_fn(l2_w, lbl_r)\n"
        "            loss = loss_h1 + loss_h2 + 0.5 * loss_pre\n"
        "        else:\n"
        "            (l1, la_hm, lb_hm, lam_hm, sc_hm, mu_hm, lv_hm, pre_hm, f_recon_hm, f_hm) = model(imgs_hm, labels=lbl_hm, mode='head_median', use_hm=use_hm)\n"
        "            (l2, la_r, lb_r, lam_r, mu_r, lv_r, pre_r, f_recon_r, f_r) = model(imgs_r, labels=lbl_r, mode='tail', use_hm=use_hm)\n"
    )
    NEW_WARMUP_FWD = (
        "        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):\n"
        "          if warmup:\n"
        "            # We use the mode 'warmup' to route through DataParallel\n"
        "            # We pass imgs_r in the labels argument slot to route it through DataParallel together\n"
        "            _, pre_logits, l1, l2_w = model(imgs_hm, labels=imgs_r, mode='warmup')\n"
        "            loss_pre = F.cross_entropy(pre_logits, lbl_hm)\n"
        "            loss_h1 = ebs_fn(l1, lbl_hm)\n"
        "            loss_h2 = ebs_fn(l2_w, lbl_r)\n"
        "            loss = loss_h1 + loss_h2 + 0.5 * loss_pre\n"
        "          else:\n"
        "            (l1, la_hm, lb_hm, lam_hm, sc_hm, mu_hm, lv_hm, pre_hm, f_recon_hm, f_hm) = model(imgs_hm, labels=lbl_hm, mode='head_median', use_hm=use_hm)\n"
        "            (l2, la_r, lb_r, lam_r, mu_r, lv_r, pre_r, f_recon_r, f_r) = model(imgs_r, labels=lbl_r, mode='tail', use_hm=use_hm)\n"
    )

    # The rest of the else block also needs to be indented inside the `with` block
    OLD_LOSS_BLOCK = (
        "            if use_hm:\n"
        "                loss_h1 = mixed_ce(l1, la_hm, lb_hm, lam_hm, ebs_fn)\n"
        "                loss_h2 = mixed_ce(l2, la_r,  lb_r,  lam_r,  ebs_fn)\n"
        "            else:\n"
        "                loss_h1 = ebs_fn(l1, lbl_hm)\n"
        "                loss_h2 = ebs_fn(l2, lbl_r)\n"
        "\n"
        "            head_mask = torch.tensor([lbl.item() in [c for c,g in CLASS_GROUPS.items() if g=='head'] for lbl in lbl_hm]).to(DEVICE)\n"
        "            loss_head = ebs_fn(l1[head_mask], lbl_hm[head_mask]) * 0.3 if head_mask.any() else torch.tensor(0.0, device=DEVICE)\n"
        "\n"
        "            loss_cvae = (cvae_loss(f_recon_hm, f_hm.detach(), mu_hm, lv_hm) + cvae_loss(f_recon_r, f_r.detach(), mu_r, lv_r)) * 0.5\n"
        "            loss_sc = sup_con(sc_hm[:len(lbl_i)], lbl_i)\n"
        "            loss_pre = (F.cross_entropy(pre_hm, lbl_hm) + F.cross_entropy(pre_r,  lbl_r)) * 0.5\n"
        "            loss = loss_h1 + loss_h2 + 0.3 * loss_sc + 0.2 * loss_cvae + 0.3 * loss_pre + loss_head\n"
        "\n"
        "        loss.backward()\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)\n"
        "        opt.step()\n"
        "        total += loss.item(); n += 1\n"
        "    return total / max(n, 1)\n"
    )
    NEW_LOSS_BLOCK = (
        "            if use_hm:\n"
        "                loss_h1 = mixed_ce(l1, la_hm, lb_hm, lam_hm, ebs_fn)\n"
        "                loss_h2 = mixed_ce(l2, la_r,  lb_r,  lam_r,  ebs_fn)\n"
        "            else:\n"
        "                loss_h1 = ebs_fn(l1, lbl_hm)\n"
        "                loss_h2 = ebs_fn(l2, lbl_r)\n"
        "\n"
        "            head_mask = torch.tensor([lbl.item() in [c for c,g in CLASS_GROUPS.items() if g=='head'] for lbl in lbl_hm]).to(DEVICE)\n"
        "            loss_head = ebs_fn(l1[head_mask], lbl_hm[head_mask]) * 0.3 if head_mask.any() else torch.tensor(0.0, device=DEVICE)\n"
        "\n"
        "            loss_cvae = (cvae_loss(f_recon_hm, f_hm.detach(), mu_hm, lv_hm) + cvae_loss(f_recon_r, f_r.detach(), mu_r, lv_r)) * 0.5\n"
        "            loss_sc = sup_con(sc_hm[:len(lbl_i)], lbl_i)\n"
        "            loss_pre = (F.cross_entropy(pre_hm, lbl_hm) + F.cross_entropy(pre_r,  lbl_r)) * 0.5\n"
        "            loss = loss_h1 + loss_h2 + 0.3 * loss_sc + 0.2 * loss_cvae + 0.3 * loss_pre + loss_head\n"
        "\n"
        "        loss.backward()\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)\n"
        "        opt.step()\n"
        "        total += loss.item(); n += 1\n"
        "    if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        "    return total / max(n, 1)\n"
    )

    if OLD_WARMUP_FWD in src:
        src = src.replace(OLD_WARMUP_FWD, NEW_WARMUP_FWD)
    if OLD_LOSS_BLOCK in src:
        src = src.replace(OLD_LOSS_BLOCK, NEW_LOSS_BLOCK)

    # ── 5. evaluate(return_all=True): fix broken bottleneck call ─────────────
    # Replace the `model.bottleneck(model.encoder(imgs))` pattern with just
    # collecting encoder features directly (encoder output is the latent).
    OLD_LATENT_DP = (
        "        if return_all:\n"
        "            probs.append(prob.cpu())\n"
        "            if isinstance(model, nn.DataParallel):\n"
        "                f = model.module.bottleneck(model.module.encoder(imgs))\n"
        "            else:\n"
        "                f = model.bottleneck(model.encoder(imgs))\n"
        "            latents.append(f.cpu())\n"
    )
    NEW_LATENT_DP = (
        "        if return_all:\n"
        "            probs.append(prob.cpu())\n"
        "            enc = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder\n"
        "            latents.append(enc(imgs).cpu())\n"
    )
    src = src.replace(OLD_LATENT_DP, NEW_LATENT_DP)

    # ── 6. gc.collect() + empty_cache after training loop ────────────────────
    OLD_AFTER_TRAIN = (
        "# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────\n"
        "import time\n"
    )
    NEW_AFTER_TRAIN = (
        "# Free GPU memory before evaluation\n"
        "import gc\n"
        "gc.collect()\n"
        "if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        "\n"
        "# ── 5. EVALUATION, METRICS, PUBLICATION FIGURES & STATS ─────────────────────\n"
        "import time\n"
    )
    src = src.replace(OLD_AFTER_TRAIN, NEW_AFTER_TRAIN)

    # ── 7. save_fig: only PNG (drop PDF/SVG) ─────────────────────────────────
    OLD_SAVE_FIG = (
        "def save_fig(name):\n"
        "    for ext in ['png', 'pdf', 'svg']:\n"
        "        plt.savefig(os.path.join(EXP_DIR, 'figures', f'{name}.{ext}'), dpi=300, bbox_inches='tight')\n"
        "    plt.close()\n"
    )
    NEW_SAVE_FIG = (
        "def save_fig(name):\n"
        "    # Save PNG only to reduce disk I/O and memory pressure on Kaggle\n"
        "    plt.savefig(os.path.join(EXP_DIR, 'figures', f'{name}.png'), dpi=150, bbox_inches='tight')\n"
        "    plt.close()\n"
    )
    src = src.replace(OLD_SAVE_FIG, NEW_SAVE_FIG)

    # ── 8. t-SNE/UMAP subsample: 5000 → 2000 ────────────────────────────────
    src = src.replace(
        "    if len(test_latents) > 5000:\n"
        "        # Subsample for speed\n"
        "        idx = np.random.choice(len(test_latents), 5000, replace=False)\n",
        "    if len(test_latents) > 2000:\n"
        "        # Subsample for speed and to avoid OOM during TSNE/UMAP\n"
        "        idx = np.random.choice(len(test_latents), 2000, replace=False)\n",
    )

    # ── 9. Free test_latents after embedding plots ────────────────────────────
    OLD_EMBED_END = (
        "except Exception as e:\n"
        "    print(f\"Embedding visualization failed: {e}\")\n"
    )
    NEW_EMBED_END = (
        "except Exception as e:\n"
        "    print(f\"Embedding visualization failed: {e}\")\n"
        "finally:\n"
        "    del test_latents\n"
        "    gc.collect()\n"
        "    if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
    )
    src = src.replace(OLD_EMBED_END, NEW_EMBED_END)

    # ── Also: zero_grad(set_to_none=True) for scripts using tqdm ─────────────
    # Some scripts have `enumerate(tqdm(loader_instance...))` – already handled above

    if src == original:
        print(f"  [SKIP – already patched or no matches] {os.path.basename(path)}")
        return False

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"  [PATCHED] {os.path.basename(path)}")
    return True


def main():
    files = [TEMPLATE]
    for fn in sorted(os.listdir(SCRIPTS_DIR)):
        if fn.endswith(".py"):
            files.append(os.path.join(SCRIPTS_DIR, fn))

    patched = 0
    for fp in files:
        if patch_file(fp):
            patched += 1

    print(f"\nDone: {patched}/{len(files)} files patched.")


if __name__ == "__main__":
    main()
