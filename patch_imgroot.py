"""
patch_imgroot.py – Replace naive img_root discovery with smart two-pass version
that prefers ImageFolder-compatible directories (class subdirs with images)
over flat image directories. Skips blood_cell_detection.py (already patched).
"""

import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")

OLD = (
    "# ── 2. DATA LOADING & SAMPLING ─────────────────────────────────────────────\r\n"
    "# Find the directory containing the actual images\r\n"
    "img_root = DATASET_PATH\r\n"
    "image_files = []\r\n"
    "for root, dirs, files in os.walk(DATASET_PATH):\r\n"
    "    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]\r\n"
    "    if len(imgs) > len(image_files):\r\n"
    "        image_files = [os.path.join(root, f) for f in imgs]\r\n"
    "        img_root = root\r\n"
    "\r\n"
    "class_dirs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]\r\n"
)

NEW = (
    "# ── 2. DATA LOADING & SAMPLING ─────────────────────────────────────────────\r\n"
    "# Find the best img_root: prefer a directory whose subdirectories each contain\r\n"
    "# images (ImageFolder layout) over a flat directory of unlabelled images.\r\n"
    "img_root = DATASET_PATH\r\n"
    "image_files = []\r\n"
    "best_flat_root = DATASET_PATH\r\n"
    "best_flat_count = 0\r\n"
    "best_clf_root = None     # directory whose subdirs contain images (class dirs)\r\n"
    "best_clf_count = 0       # number of such image-bearing subdirs\r\n"
    "\r\n"
    "for root, dirs, files in os.walk(DATASET_PATH):\r\n"
    "    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]\r\n"
    "    # Track the flat directory with the most images (fallback)\r\n"
    "    if len(imgs) > best_flat_count:\r\n"
    "        best_flat_count = len(imgs)\r\n"
    "        best_flat_root = root\r\n"
    "\r\n"
    "    # Check if this directory has subdirs that each contain images (class dirs)\r\n"
    "    clf_subs = []\r\n"
    "    for d in dirs:\r\n"
    "        sub_path = os.path.join(root, d)\r\n"
    "        try:\r\n"
    "            sub_imgs = [f for f in os.listdir(sub_path)\r\n"
    "                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]\r\n"
    "        except PermissionError:\r\n"
    "            continue\r\n"
    "        if len(sub_imgs) > 0:\r\n"
    "            clf_subs.append(d)\r\n"
    "    if len(clf_subs) >= 2 and len(clf_subs) > best_clf_count:\r\n"
    "        best_clf_count = len(clf_subs)\r\n"
    "        best_clf_root = root\r\n"
    "\r\n"
    "# Prefer ImageFolder-compatible root; fall back to flat directory\r\n"
    "if best_clf_root is not None:\r\n"
    "    img_root = best_clf_root\r\n"
    "    print(f\"Found ImageFolder-compatible root with {best_clf_count} class dirs: {img_root}\")\r\n"
    "else:\r\n"
    "    img_root = best_flat_root\r\n"
    "    print(f\"No class subdirs found; using flat image root: {img_root}\")\r\n"
    "\r\n"
    "# Rebuild image_files from the chosen root\r\n"
    "image_files = []\r\n"
    "for root2, _, files2 in os.walk(img_root):\r\n"
    "    for f in files2:\r\n"
    "        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):\r\n"
    "            image_files.append(os.path.join(root2, f))\r\n"
    "\r\n"
    "class_dirs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]\r\n"
)


def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if OLD not in src:
        # Already patched or uses different structure
        print(f"  [SKIP] {os.path.basename(path)}")
        return False

    src = src.replace(OLD, NEW)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"  [PATCHED] {os.path.basename(path)}")
    return True


def main():
    files = []
    for fn in sorted(os.listdir(SCRIPTS_DIR)):
        if fn.endswith(".py") and fn != "run_blood_cell_detection.py":
            files.append(os.path.join(SCRIPTS_DIR, fn))

    patched = 0
    for fp in files:
        if patch_file(fp):
            patched += 1

    print(f"\nDone: {patched}/{len(files)} files patched.")


if __name__ == "__main__":
    main()
