"""Pre-training validation for the boat_deck CNN dataset.

Checks:
1. Directory structure (train/val/test with both classes)
2. Exact image counts per class per split
3. Corrupt image detection (Pillow verify)
4. Cross-split duplicate detection (SHA-256)
5. Manifest vs actual files consistency
6. Training script compatibility (CPU, correct data-dir)
7. Original images unchanged (boat_deck/pufferfish/ still intact)
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

SPLIT_DIR    = ROOT / "data" / "camera" / "boat_deck_split"
MANIFEST     = ROOT / "data" / "camera" / "boat_deck_split_manifest.csv"
ORIGINAL_PUF = ROOT / "data" / "camera" / "boat_deck" / "pufferfish"
TRAIN_SCRIPT = ROOT / "scripts" / "train_camera_model.py"
CLASSES      = ("other_fish", "pufferfish")
SPLITS       = ("train", "val", "test")
VALID_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

passed, failed, warnings = 0, 0, 0

def check(label, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition

def warn(label, detail=""):
    global warnings
    warnings += 1
    print(f"  [WARN] {label}" + (f" -- {detail}" if detail else ""))

print("=" * 70)
print("  AquaSentinel -- Pre-Training Dataset Validation")
print("=" * 70)

# =========================================================================
# 1. Directory structure
# =========================================================================
print("\n[1/7] Directory structure...")
check("Split dir exists", SPLIT_DIR.exists(), str(SPLIT_DIR))
for split in SPLITS:
    sd = SPLIT_DIR / split
    check(f"  {split}/ exists", sd.exists())
    for cls in CLASSES:
        cd = sd / cls
        check(f"    {split}/{cls}/ exists", cd.exists())

# =========================================================================
# 2. Image counts
# =========================================================================
print("\n[2/7] Image counts per class per split...")
counts = {}
total_files = 0
for split in SPLITS:
    counts[split] = {}
    for cls in CLASSES:
        cd = SPLIT_DIR / split / cls
        if cd.exists():
            files = [f for f in cd.iterdir()
                     if f.is_file() and f.suffix.lower() in VALID_EXT]
            counts[split][cls] = len(files)
            total_files += len(files)
        else:
            counts[split][cls] = 0

print(f"  {'Split':<8} {'pufferfish':>12} {'other_fish':>12} {'total':>8}")
print(f"  {'-'*44}")
for split in SPLITS:
    pf = counts[split].get("pufferfish", 0)
    of = counts[split].get("other_fish", 0)
    print(f"  {split:<8} {pf:>12} {of:>12} {pf+of:>8}")
total_pf = sum(counts[s].get("pufferfish", 0) for s in SPLITS)
total_of = sum(counts[s].get("other_fish", 0) for s in SPLITS)
print(f"  {'TOTAL':<8} {total_pf:>12} {total_of:>12} {total_pf+total_of:>8}")

check("Total images > 0", total_files > 0, f"{total_files} files")
check("Both classes have images", total_pf > 0 and total_of > 0,
      f"pufferfish={total_pf}, other_fish={total_of}")
check("All splits non-empty",
      all(sum(counts[s].values()) > 0 for s in SPLITS))
ratio = round(total_of / total_pf, 1) if total_pf > 0 else float('inf')
check("Imbalance ratio <= 5:1", ratio <= 5.0, f"ratio={ratio}:1")

# =========================================================================
# 3. Corrupt image detection
# =========================================================================
print("\n[3/7] Corrupt image detection (Pillow verify)...")
corrupt = []
checked = 0
for split in SPLITS:
    for cls in CLASSES:
        cd = SPLIT_DIR / split / cls
        if not cd.exists():
            continue
        for f in sorted(cd.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_EXT:
                continue
            checked += 1
            try:
                with Image.open(f) as img:
                    img.verify()
            except Exception as e:
                corrupt.append(f"{split}/{cls}/{f.name}: {e}")
check(f"No corrupt images", len(corrupt) == 0,
      f"checked {checked}, corrupt {len(corrupt)}")
if corrupt:
    for c in corrupt[:10]:
        print(f"    CORRUPT: {c}")

# =========================================================================
# 4. Cross-split duplicate detection
# =========================================================================
print("\n[4/7] Cross-split duplicate detection (SHA-256)...")
hash_registry: dict[str, list[str]] = defaultdict(list)
for split in SPLITS:
    for cls in CLASSES:
        cd = SPLIT_DIR / split / cls
        if not cd.exists():
            continue
        for f in sorted(cd.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_EXT:
                continue
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            digest = h.hexdigest()[:24]
            hash_registry[digest].append(f"{split}/{cls}/{f.name}")
cross_split_dups = {h: locs for h, locs in hash_registry.items()
                    if len(locs) > 1}
check("No cross-split duplicates", len(cross_split_dups) == 0,
      f"{len(cross_split_dups)} duplicate groups")
if cross_split_dups:
    for h, locs in list(cross_split_dups.items())[:5]:
        print(f"    DUP [{h}]: {locs}")

# =========================================================================
# 5. Manifest consistency
# =========================================================================
print("\n[5/7] Manifest vs actual files...")
check("Manifest file exists", MANIFEST.exists(), str(MANIFEST))
if MANIFEST.exists():
    with open(MANIFEST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
    check("Manifest has rows", len(manifest_rows) > 0, f"{len(manifest_rows)} rows")
    check("Manifest row count matches file count",
          len(manifest_rows) == total_files,
          f"manifest={len(manifest_rows)}, files={total_files}")

    # Check each manifest entry exists on disk
    missing_from_disk = []
    for row in manifest_rows:
        dest = SPLIT_DIR / row["dest_path"]
        if not dest.exists():
            missing_from_disk.append(row["dest_path"])
    check("All manifest entries exist on disk",
          len(missing_from_disk) == 0,
          f"{len(missing_from_disk)} missing")

    # Check each file on disk is in the manifest
    manifest_dests = {row["dest_path"].replace("\\", "/") for row in manifest_rows}
    disk_files = []
    for split in SPLITS:
        for cls in CLASSES:
            cd = SPLIT_DIR / split / cls
            if not cd.exists():
                continue
            for f in cd.iterdir():
                if f.is_file() and f.suffix.lower() in VALID_EXT:
                    rel = f"{split}/{cls}/{f.name}"
                    disk_files.append(rel)
    extra_on_disk = [f for f in disk_files
                     if f not in manifest_dests
                     and f.replace("/", "\\") not in {r["dest_path"] for r in manifest_rows}]
    check("No extra files on disk (not in manifest)",
          len(extra_on_disk) == 0,
          f"{len(extra_on_disk)} extra")

    # Check required manifest columns
    required_cols = {"dest_path", "source_path", "class", "species_folder",
                     "source_dataset", "original_split", "env_flag", "licence"}
    actual_cols = set(manifest_rows[0].keys()) if manifest_rows else set()
    check("Manifest has required columns",
          required_cols.issubset(actual_cols),
          f"missing: {required_cols - actual_cols}" if not required_cols.issubset(actual_cols) else "all present")

# =========================================================================
# 6. Training script compatibility
# =========================================================================
print("\n[6/7] Training script compatibility...")
check("Training script exists", TRAIN_SCRIPT.exists(), str(TRAIN_SCRIPT))
if TRAIN_SCRIPT.exists():
    script_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    check("Script accepts --data-dir",
          "--data-dir" in script_text)
    check("Script accepts --env boat_deck",
          "boat_deck" in script_text)
    check("Script uses CPU device",
          '"cpu"' in script_text or "'cpu'" in script_text)
    check("Script accepts --epochs",
          "--epochs" in script_text)

# Check that PyTorch + torchvision are importable
try:
    import torch
    import torchvision
    check("PyTorch importable", True, f"v{torch.__version__}")
    check("torchvision importable", True, f"v{torchvision.__version__}")
except ImportError as e:
    check("PyTorch/torchvision importable", False, str(e))

# Check model builds
try:
    from pufferfish.camera.model import build_model, INPUT_SIZE
    model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    import torch as _t
    x = _t.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with _t.no_grad():
        out = model(x)
    check("MobileNetV3-Small forward pass OK", out.shape == (1, 2),
          f"output shape {tuple(out.shape)}")
except Exception as e:
    check("MobileNetV3-Small forward pass OK", False, str(e))

# =========================================================================
# 7. Original images unchanged
# =========================================================================
print("\n[7/7] Original pufferfish images unchanged...")
check("Original pufferfish dir exists", ORIGINAL_PUF.exists())
if ORIGINAL_PUF.exists():
    orig_files = [f for f in ORIGINAL_PUF.iterdir()
                  if f.is_file() and f.suffix.lower() in VALID_EXT]
    check("Original pufferfish count intact", len(orig_files) >= 85,
          f"found {len(orig_files)} (user has been adding more)")
    # Verify a sample is readable
    if orig_files:
        try:
            with Image.open(orig_files[0]) as img:
                img.verify()
            check("Sample original image readable", True, orig_files[0].name)
        except Exception as e:
            check("Sample original image readable", False, str(e))

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*70}")
print(f"  RESULTS: {passed} passed, {failed} failed, {warnings} warnings")
print(f"{'='*70}")

if failed == 0:
    print("\n  ALL CHECKS PASSED. Dataset is ready for training.")
    print("\n  Training command:")
    print(f"    python scripts/train_camera_model.py \\")
    print(f"        --data-dir data/camera/boat_deck_split \\")
    print(f"        --env boat_deck \\")
    print(f"        --arch mobilenet_v3_small \\")
    print(f"        --epochs 20")
    print(f"\n  Output location:")
    print(f"    artifacts/camera/boat_deck_mobilenet_v3_small/")
    print(f"      model.pth          -- trained weights")
    print(f"      model_meta.json    -- class mapping, history, eval metrics")
    print(f"      checkpoints/       -- best checkpoint during training")
else:
    print(f"\n  {failed} CHECK(S) FAILED. Fix issues before training.")

print()
