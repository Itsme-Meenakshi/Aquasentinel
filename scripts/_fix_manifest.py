"""Remove manifest entries for files that no longer exist on disk."""
import csv
from pathlib import Path

split_dir = Path("data/camera/boat_deck_split")
manifest  = Path("data/camera/boat_deck_split_manifest.csv")

with open(manifest, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

kept = []
removed = 0
for row in rows:
    dest = split_dir / row["dest_path"]
    if dest.exists():
        kept.append(row)
    else:
        removed += 1
        dp = row["dest_path"]
        print(f"  Removed: {dp}")

with open(manifest, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(kept)

print(f"\nManifest: {len(rows)} -> {len(kept)} rows ({removed} removed)")
