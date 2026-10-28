"""Train the pufferfish camera recognition model.

Usage
-----
  python scripts/train_camera_model.py \\
      --data-dir data/camera/boat_deck \\
      --env boat_deck \\
      --arch mobilenet_v3_small \\
      --epochs 20

The script expects data_dir/ to have the split structure:
  data_dir/train/pufferfish/, data_dir/train/other_fish/
  data_dir/val/pufferfish/,   data_dir/val/other_fish/
  data_dir/test/pufferfish/,  data_dir/test/other_fish/

If the split/ subdirectories don't exist, run --split-raw first.

Available environments: boat_deck, underwater
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pufferfish.camera.dataset import (
    CLASS_NAMES, scan_directory, split_dataset, class_balance_report
)
from pufferfish.camera.model import (
    build_model, create_dataloaders, train_model, evaluate_model,
    save_model_artifacts
)


def parse_args():
    p = argparse.ArgumentParser(description="Train pufferfish camera model")
    p.add_argument("--data-dir", required=True,
                   help="Root data directory (must contain train/val/test class subdirs)")
    p.add_argument("--env", default="boat_deck",
                   choices=["boat_deck", "underwater"],
                   help="Dataset environment (kept separate for each model)")
    p.add_argument("--arch", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "resnet18"],
                   help="Model architecture")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir",
                   help="Where to save model artifacts "
                        "(default: artifacts/camera/<env>_<arch>/)")
    p.add_argument("--split-raw",
                   help="Path to raw (unsplit) image dir to split first. "
                        "After splitting, files are moved to --data-dir.")
    p.add_argument("--notes", default="", help="Notes to include in model metadata")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "artifacts" / "camera" / f"{args.env}_{args.arch}"
    )

    print(f"\n{'='*60}")
    print(f"  AquaSentinel — Camera Model Training")
    print(f"  Environment : {args.env}")
    print(f"  Architecture: {args.arch}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*60}\n")

    # Optional: split raw data first
    if args.split_raw:
        raw_dir = Path(args.split_raw)
        print(f"[1/5] Scanning raw data directory: {raw_dir}")
        scan_report = scan_directory(raw_dir)
        print(f"      Valid images: {scan_report['valid_images']}")
        for cls in CLASS_NAMES:
            count = scan_report["classes"].get(cls, {}).get("count", 0)
            print(f"      {cls}: {count}")
        if scan_report["invalid_images"]:
            print(f"  WARNING: {len(scan_report['invalid_images'])} invalid files")
        print(f"[2/5] Splitting into train/val/test → {data_dir}")
        splits = split_dataset(raw_dir, data_dir)
        print(f"      Train: {splits['train']}")
        print(f"      Val:   {splits['val']}")
        print(f"      Test:  {splits['test']}")
    else:
        print("[1/5] Using pre-split data directory")
        print("[2/5] Skipped (no --split-raw)")

    # Class balance check
    print("[3/5] Checking class balance...")
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if split_dir.exists():
            bal = class_balance_report(split_dir)
            print(f"      {split}: {bal['counts']} | ratio={bal['imbalance_ratio']}")
            if bal["imbalance_ratio"] > 5:
                print(f"  WARNING: High imbalance in {split} split!")

    # Build model
    print(f"[4/5] Building {args.arch} (pretrained={'No' if args.no_pretrained else 'ImageNet'})...")
    model = build_model(
        arch=args.arch,
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained
    )

    # Create dataloaders
    loaders = create_dataloaders(data_dir, batch_size=args.batch_size)
    if "train" not in loaders:
        print(f"ERROR: No train/ subdir found in {data_dir}. "
              f"Run with --split-raw to create splits, or check directory structure.")
        sys.exit(1)
    print(f"      Loaders: {list(loaders.keys())}")

    # Train
    print(f"\n[5/5] Training for up to {args.epochs} epochs...")
    checkpoint_dir = output_dir / "checkpoints"
    history = train_model(
        model=model,
        dataloaders=loaders,
        num_epochs=args.epochs,
        lr=args.lr,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )

    # Load best checkpoint if available
    best_ckpt = checkpoint_dir / "best_model.pth"
    if best_ckpt.exists():
        import torch
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu", weights_only=True))
        print(f"  Loaded best checkpoint (epoch {history.get('best_epoch', '?')+1})")

    # Evaluate on test set
    eval_results = None
    if "test" in loaders:
        print("\n  Evaluating on test set...")
        eval_results = evaluate_model(model, loaders["test"], device="cpu")
        print(f"  Accuracy:    {eval_results['accuracy']:.4f}")
        print(f"  Macro F1:    {eval_results['macro_f1']:.4f}")
        for cls in CLASS_NAMES:
            pc = eval_results["per_class"][cls]
            print(f"  {cls}: P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}")

    # Save artifacts
    save_model_artifacts(
        model=model,
        history=history,
        eval_results=eval_results,
        arch=args.arch,
        environment=args.env,
        output_dir=output_dir,
        notes=args.notes,
    )
    print(f"\n  Model saved to: {output_dir}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
