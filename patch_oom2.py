"""
patch_oom2.py  –  Add pin_memory + prefetch_factor to DataLoader calls.

The first patch missed these because the DataLoader calls don't end with )
on the same line. This patch does a direct string replacement.
"""

import os

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
TEMPLATE    = os.path.join(os.path.dirname(__file__), "script_template.py")

REPLACEMENTS = [
    (
        "loader_instance = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=instance_sampler(train_labels), num_workers=NUM_WORKERS)\n"
        "loader_median   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=median_sampler(train_labels), num_workers=NUM_WORKERS)\n"
        "loader_reverse  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=reverse_sampler(train_labels), num_workers=NUM_WORKERS)\n"
        "val_loader      = DataLoader(AugmentedSubset(val_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)\n"
        "test_loader     = DataLoader(AugmentedSubset(test_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)\n",
        # Replacement: add pin_memory + prefetch_factor
        "loader_instance = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=instance_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)\n"
        "loader_median   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=median_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)\n"
        "loader_reverse  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=reverse_sampler(train_labels), num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)\n"
        "val_loader      = DataLoader(AugmentedSubset(val_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)\n"
        "test_loader     = DataLoader(AugmentedSubset(test_ds, transform=eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)\n",
    ),
]


def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    original = src
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)

    if src == original:
        print(f"  [SKIP] {os.path.basename(path)}")
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
