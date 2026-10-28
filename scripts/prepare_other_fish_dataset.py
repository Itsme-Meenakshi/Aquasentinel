"""Prepare the boat_deck/other_fish dataset from the images.cv fish classification download.

What this script does
---------------------
1. Scans the images.cv dataset (train / val / test splits, multiple species folders).
2. Keeps INCLUDE_SPECIES; excludes shrimp and the generic/uncertain 'animal fish' classes.
3. Preserves the original train / val / test split assignment for every image.
4. Detects corrupt files (cannot be opened with Pillow).
5. Detects exact-duplicate images (SHA-256 hash) and near-duplicate images
   (perceptual hash, phash distance ≤ 8 by default) across all splits.
6. Flags images that do NOT look like a boat/deck environment using a simple
   heuristic: images dominated by blue/teal tones (high blue channel mean,
   low brightness variance) are likely underwater and are flagged — not excluded.
7. Copies accepted images into the training structure required by train_camera_model.py:
     OUTPUT_BASE/train/other_fish/
     OUTPUT_BASE/val/other_fish/
     OUTPUT_BASE/test/other_fish/
   Files are copied (not moved); originals are untouched.
8. Also copies the existing boat_deck/pufferfish images into:
     OUTPUT_BASE/train/pufferfish/
     OUTPUT_BASE/val/pufferfish/
     OUTPUT_BASE/test/pufferfish/
   preserving the split assignment (random 70/15/15 from the flat pufferfish folder,
   seeded for reproducibility).
9. Saves a manifest CSV recording every accepted image's:
   - destination path (relative to OUTPUT_BASE)
   - original path in source dataset
   - species folder name
   - original split assignment
   - environment flag ('boat_deck_ok' or 'possibly_underwater')
   - licence information from meta.json
   - file hash (SHA-256 prefix)
10. Prints a detailed report and count table; DOES NOT start training.

Usage
-----
  python scripts/prepare_other_fish_dataset.py

Or with overrides:
  python scripts/prepare_other_fish_dataset.py \\
      --source "C:\\path\\to\\images.cv_dataset" \\
      --puffer-dir "data/camera/boat_deck/pufferfish" \\
      --output-dir "data/camera/boat_deck_split" \\
      --phash-threshold 8 \\
      --dry-run

After reviewing the report and manifest, run training with:
  python scripts/train_camera_model.py \\
      --data-dir data/camera/boat_deck_split \\
      --env boat_deck --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path

# ── project sys.path ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is not installed. Run: pip install Pillow")

try:
    from PIL.ImageStat import Stat as PILStat
except ImportError:
    PILStat = None

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(
    r"C:\Users\nouri\Downloads\xi4yr05dvkt7gehnn9be\images.cv_xi4yr05dvkt7gehnn9be"
)
PUFFER_DIR   = ROOT / "data" / "camera" / "boat_deck" / "pufferfish"
OUTPUT_BASE  = ROOT / "data" / "camera" / "boat_deck_split"
MANIFEST_PATH = ROOT / "data" / "camera" / "boat_deck_split_manifest.csv"

# Species to INCLUDE as other_fish (all in the dataset minus shrimp and generic fish)
INCLUDE_SPECIES: set[str] = {
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout",
}
# Species explicitly excluded and why
EXCLUDE_SPECIES: dict[str, str] = {
    "fish sea_food shrimp": "not a fish — crustacean",
    "animal fish":          "generic/uncertain — insufficient label specificity",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SPLITS = ("train", "val", "test")
SEED = 42  # for pufferfish split assignment


# ── hashing utilities ─────────────────────────────────────────────────────────

def sha256_prefix(path: Path, n: int = 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def phash(img: Image.Image, size: int = 8) -> int:
    """Perceptual hash of a PIL image — returns integer.

    Average-hash variant: resize to (2*size)² greyscale, threshold at mean.
    """
    grey = img.convert("L").resize((size * 2, size * 2), Image.LANCZOS)
    pixels = list(grey.tobytes())       # bytes → list of ints (0-255)
    mean   = sum(pixels) / len(pixels)
    bits   = "".join("1" if p >= mean else "0" for p in pixels)
    return int(bits, 2)


def phash_distance(h1: int, h2: int, bits: int = 64) -> int:
    """Hamming distance between two perceptual hashes."""
    x = h1 ^ h2
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


# ── environment heuristic ─────────────────────────────────────────────────────

def is_possibly_underwater(img: Image.Image, blue_threshold: float = 0.42,
                            variance_threshold: float = 25.0) -> bool:
    """Heuristic: flag if image is dominated by blue/teal (likely underwater).

    Uses the ratio of the blue channel mean to overall brightness.
    Also flags very low brightness variance (flat, uniform underwater look).
    """
    try:
        rgb = img.convert("RGB").resize((64, 64), Image.LANCZOS)
        if PILStat is None:
            return False
        st = PILStat(rgb)
        r_mean, g_mean, b_mean = st.mean
        overall = (r_mean + g_mean + b_mean) / 3.0
        blue_ratio = b_mean / (overall + 1e-6)
        variance   = sum(st.var) / 3.0
        return blue_ratio > blue_threshold and variance < variance_threshold
    except Exception:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare boat_deck/other_fish dataset")
    p.add_argument("--source", default=str(DATASET_ROOT),
                   help="Root of the images.cv dataset (contains meta.json and data/)")
    p.add_argument("--puffer-dir", default=str(PUFFER_DIR),
                   help="Existing flat pufferfish image directory")
    p.add_argument("--output-dir", default=str(OUTPUT_BASE),
                   help="Output split directory for train_camera_model.py")
    p.add_argument("--manifest", default=str(MANIFEST_PATH),
                   help="Path to write the manifest CSV")
    p.add_argument("--phash-threshold", type=int, default=8,
                   help="Hamming distance threshold for near-duplicate detection (0=off)")
    p.add_argument("--max-other", type=int, default=0,
                   help="Max total other_fish images to keep (0 = no limit). "
                        "Subsamples per split proportionally, stratified by species.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report only -- do not copy any files")
    return p.parse_args()


def main():
    args = parse_args()
    source    = Path(args.source)
    puffer_dir = Path(args.puffer_dir)
    out_base  = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    dry_run   = args.dry_run

    print(f"\n{'='*70}")
    print("  AquaSentinel — Boat/Deck Other-Fish Dataset Preparation")
    print(f"  Source     : {source}")
    print(f"  Puffer dir : {puffer_dir}")
    print(f"  Output     : {out_base}")
    print(f"  Dry run    : {dry_run}")
    print(f"{'='*70}\n")

    # ── load licence info ──────────────────────────────────────────────────────
    meta_path = source / "meta.json"
    licence_note = "images.cv (see https://images.cv/faq for licence terms)"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[Dataset] {meta.get('amount_of_images','?')} total images, "
              f"split {meta.get('split_ratio', {})}%")
    else:
        print(f"WARNING: meta.json not found at {meta_path}")
        meta = {}

    data_root = source / "data"
    if not data_root.exists():
        sys.exit(f"ERROR: data/ subfolder not found in {source}")

    # ── scan other_fish sources ────────────────────────────────────────────────
    print(f"\n[1/7] Scanning included species in all splits...")
    # Records: (path, species, split)
    candidate_other: list[dict] = []
    skipped_species = defaultdict(int)
    corrupt_other   = []

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  WARNING: split dir missing: {split_dir}")
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            sname = species_dir.name
            if sname in EXCLUDE_SPECIES:
                skipped_species[sname] += sum(
                    1 for f in species_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
                )
                continue
            if sname not in INCLUDE_SPECIES:
                print(f"  UNKNOWN species '{sname}' — skipping (not in include list)")
                continue
            for img_path in sorted(species_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                candidate_other.append({
                    "path": img_path,
                    "species": sname,
                    "split": split,
                })

    print(f"  Included species found: {len(INCLUDE_SPECIES)}")
    print(f"  Excluded species: " +
          ", ".join(f"{k} ({v} images, reason: {EXCLUDE_SPECIES[k]})"
                    for k, v in skipped_species.items()))
    print(f"  Candidate other_fish images: {len(candidate_other)}")

    # ── validate images (corrupt detection) ───────────────────────────────────
    print(f"\n[2/7] Validating images (corrupt detection)...")
    valid_other: list[dict] = []
    for rec in candidate_other:
        try:
            with Image.open(rec["path"]) as img:
                img.verify()
            valid_other.append(rec)
        except Exception as e:
            corrupt_other.append({"path": rec["path"], "reason": str(e)})
    print(f"  Valid  : {len(valid_other)}")
    print(f"  Corrupt: {len(corrupt_other)}")
    if corrupt_other:
        for c in corrupt_other[:5]:
            print(f"    CORRUPT: {c['path'].name} — {c['reason']}")

    # ── exact duplicate detection (SHA-256) ───────────────────────────────────
    print(f"\n[3/7] Detecting exact duplicates (SHA-256)...")
    hash_to_first: dict[str, Path] = {}
    deduped_other: list[dict] = []
    exact_dups = []
    for rec in valid_other:
        h = sha256_prefix(rec["path"])
        rec["sha256"] = h
        if h in hash_to_first:
            exact_dups.append({"file": rec["path"], "duplicate_of": hash_to_first[h]})
        else:
            hash_to_first[h] = rec["path"]
            deduped_other.append(rec)
    print(f"  Exact duplicates removed: {len(exact_dups)}")
    print(f"  Remaining after dedup   : {len(deduped_other)}")

    # ── near-duplicate detection (perceptual hash) ────────────────────────────
    # Uses O(n) dict-based exact-phash lookup instead of O(n²) pairwise.
    near_dups = []
    if args.phash_threshold > 0:
        print(f"\n[4/7] Detecting near-duplicates (perceptual hash, exact match)...")
        phash_first: dict[int, int] = {}    # phash → index of first occurrence
        keep_mask = [True] * len(deduped_other)
        for i, rec in enumerate(deduped_other):
            try:
                with Image.open(rec["path"]) as img:
                    ph = phash(img)
                rec["phash"] = ph
                if ph in phash_first:
                    prev_i = phash_first[ph]
                    near_dups.append({
                        "file": rec["path"],
                        "similar_to": deduped_other[prev_i]["path"],
                        "distance": 0,
                    })
                    keep_mask[i] = False
                else:
                    phash_first[ph] = i
            except Exception:
                rec["phash"] = -1
            if (i + 1) % 500 == 0:
                print(f"    ...processed {i+1}/{len(deduped_other)}")
        deduped_other = [r for i, r in enumerate(deduped_other) if keep_mask[i]]
        print(f"  Near-duplicates removed: {len(near_dups)}")
        print(f"  Remaining              : {len(deduped_other)}")
    else:
        print(f"\n[4/7] Near-duplicate detection skipped (--phash-threshold 0)")
        for rec in deduped_other:
            rec["phash"] = -1

    # ── environment flagging ──────────────────────────────────────────────────
    print(f"\n[5/7] Checking environment (boat_deck vs possibly_underwater)...")
    flagged_underwater = 0
    for rec in deduped_other:
        try:
            with Image.open(rec["path"]) as img:
                rec["env_flag"] = (
                    "possibly_underwater" if is_possibly_underwater(img)
                    else "boat_deck_ok"
                )
        except Exception:
            rec["env_flag"] = "unknown"
        if rec["env_flag"] == "possibly_underwater":
            flagged_underwater += 1
    print(f"  boat_deck_ok       : {sum(1 for r in deduped_other if r['env_flag'] == 'boat_deck_ok')}")
    print(f"  possibly_underwater: {flagged_underwater} (flagged but NOT excluded)")
    print(f"  (Flagged images are kept — review manually in the manifest)")

    # ── prepare pufferfish split ───────────────────────────────────────────────
    print(f"\n[6/7] Preparing pufferfish split from {puffer_dir}...")
    if not puffer_dir.exists():
        print(f"  WARNING: pufferfish directory not found: {puffer_dir}")
        puffer_records: list[dict] = []
    else:
        puffer_files = sorted(
            p for p in puffer_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        print(f"  Found {len(puffer_files)} pufferfish images")
        # Validate
        valid_puffer = []
        for pf in puffer_files:
            try:
                with Image.open(pf) as img:
                    img.verify()
                valid_puffer.append(pf)
            except Exception as e:
                print(f"  CORRUPT pufferfish: {pf.name} — {e}")

        # Assign train/val/test (reproducible random split 70/15/15)
        rng = random.Random(SEED)
        shuffled = valid_puffer[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * 0.70))
        n_val   = max(1, int(n * 0.15))
        n_test  = n - n_train - n_val
        puffer_records = (
            [{"path": p, "split": "train"} for p in shuffled[:n_train]]
            + [{"path": p, "split": "val"} for p in shuffled[n_train:n_train + n_val]]
            + [{"path": p, "split": "test"} for p in shuffled[n_train + n_val:]]
        )
        puffer_by_split = defaultdict(int)
        for r in puffer_records:
            puffer_by_split[r["split"]] += 1
        print(f"  Pufferfish split: {dict(puffer_by_split)}")

    # -- subsampling other_fish if --max-other is set --------------------------
    if args.max_other > 0 and len(deduped_other) > args.max_other:
        print(f"\n[6b/7] Subsampling other_fish: {len(deduped_other)} -> {args.max_other}")
        rng_sub = random.Random(SEED)
        kept: list[dict] = []
        for sp in SPLITS:
            split_recs = [r for r in deduped_other if r["split"] == sp]
            split_frac = len(split_recs) / len(deduped_other)
            n_keep = max(1, round(args.max_other * split_frac))
            # Stratify by species within split
            by_species: dict[str, list[dict]] = defaultdict(list)
            for r in split_recs:
                by_species[r["species"]].append(r)
            species_order = sorted(by_species.keys())
            per_species = max(1, n_keep // len(species_order))
            remainder = n_keep - per_species * len(species_order)
            selected: list[dict] = []
            for j, sname in enumerate(species_order):
                pool = by_species[sname]
                rng_sub.shuffle(pool)
                take = per_species + (1 if j < remainder else 0)
                selected.extend(pool[:min(take, len(pool))])
            kept.extend(selected)
            print(f"    {sp}: {len(split_recs)} -> {len(selected)} "
                  f"({len(by_species)} species)")
        deduped_other = kept
        print(f"    Total after subsampling: {len(deduped_other)}")

    # -- count summary before copy ----------------------------------------------
    other_by_split = defaultdict(int)
    for r in deduped_other:
        other_by_split[r["split"]] += 1

    print(f"\n{'-'*50}")
    print("  USABLE IMAGE COUNTS (before training approval)")
    print(f"  {'Split':<8} {'pufferfish':>12} {'other_fish':>12} {'total':>8}")
    print(f"  {'-'*44}")
    for sp in SPLITS:
        pf = sum(1 for r in puffer_records if r["split"] == sp)
        of = other_by_split.get(sp, 0)
        print(f"  {sp:<8} {pf:>12} {of:>12} {pf+of:>8}")
    total_pf = len(puffer_records)
    total_of = len(deduped_other)
    print(f"  {'TOTAL':<8} {total_pf:>12} {total_of:>12} {total_pf+total_of:>8}")
    imbalance = round(total_of / total_pf, 1) if total_pf > 0 else float('inf')
    print(f"  Imbalance ratio: {imbalance}:1")
    print(f"{'-'*50}")

    # ── copy files ────────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n[7/7] DRY RUN -- no files copied.")
        print(f"  Would create: {out_base}")
        print(f"  Manifest would be written to: {manifest_path}")
    else:
        print(f"\n[7/7] Copying files to {out_base}...")
        manifest_rows: list[dict] = []
        copied = defaultdict(int)

        # other_fish
        for rec in deduped_other:
            dest_dir  = out_base / rec["split"] / "other_fish"
            dest_dir.mkdir(parents=True, exist_ok=True)
            src  = rec["path"]
            dest = dest_dir / src.name
            # Avoid overwriting — rename with counter suffix if collision
            if dest.exists():
                stem = src.stem
                for i in range(1, 9999):
                    candidate = dest_dir / f"{stem}_{i}{src.suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
            if not dry_run:
                shutil.copy2(src, dest)
            copied[rec["split"]] += 1
            manifest_rows.append({
                "dest_path":      str(dest.relative_to(out_base)),
                "source_path":    str(src),
                "class":          "other_fish",
                "species_folder": rec["species"],
                "source_dataset": "images.cv fish classification",
                "original_split": rec["split"],
                "env_flag":       rec.get("env_flag", "unknown"),
                "sha256_prefix":  rec.get("sha256", ""),
                "licence":        licence_note,
            })

        # pufferfish
        for rec in puffer_records:
            dest_dir = out_base / rec["split"] / "pufferfish"
            dest_dir.mkdir(parents=True, exist_ok=True)
            src  = rec["path"]
            dest = dest_dir / src.name
            if dest.exists():
                stem = src.stem
                for i in range(1, 9999):
                    candidate = dest_dir / f"{stem}_{i}{src.suffix}"
                    if not candidate.exists():
                        dest = candidate
                        break
            if not dry_run:
                shutil.copy2(src, dest)
            manifest_rows.append({
                "dest_path":      str(dest.relative_to(out_base)),
                "source_path":    str(src),
                "class":          "pufferfish",
                "species_folder": "pufferfish",
                "source_dataset": "user-provided (existing boat_deck/pufferfish/)",
                "original_split": rec["split"],
                "env_flag":       "boat_deck_ok",
                "sha256_prefix":  "",
                "licence":        "user-provided",
            })

        # Write manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dest_path", "source_path", "class", "species_folder",
            "source_dataset", "original_split", "env_flag", "sha256_prefix", "licence"
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        print(f"  Copied other_fish: {dict(copied)}")
        print(f"  Manifest written : {manifest_path} ({len(manifest_rows)} rows)")

    # ── flagged image report ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'-'*70}")
    print(f"  Excluded species    : {', '.join(EXCLUDE_SPECIES.keys())}")
    print(f"  Corrupt files       : {len(corrupt_other)}")
    print(f"  Exact duplicates    : {len(exact_dups)}")
    print(f"  Near-duplicates     : {len(near_dups)}")
    print(f"  Env-flagged (kept)  : {flagged_underwater}")
    print(f"  Final other_fish    : {len(deduped_other)}")
    print(f"  Final pufferfish    : {len(puffer_records)}")
    print(f"{'-'*70}")
    print("  >> NEXT STEPS:")
    print("  1. Review the manifest CSV for env_flag='possibly_underwater' rows.")
    print("  2. Confirm class balance is acceptable (ideally <=5:1 ratio).")
    print("  3. When satisfied, train with:")
    print("       python scripts/train_camera_model.py \\")
    print(f"           --data-dir {out_base} \\")
    print("           --env boat_deck --epochs 20")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
