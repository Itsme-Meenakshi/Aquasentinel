# Camera Recognition — Image Data Directory

Place your training images here using this structure:

```
data/camera/
├── boat_deck/
│   ├── pufferfish/       ← photos taken on-deck or on the boat
│   │   └── img001.jpg
│   └── other_fish/
│       └── img002.jpg
│
└── underwater/
    ├── pufferfish/       ← underwater footage / GoPro frames
    │   └── img001.jpg
    └── other_fish/
        └── img002.jpg
```

## Rules

1. **Keep environments separate.** `boat_deck` and `underwater` models are trained
   and evaluated independently. Do NOT place boat images in the underwater folder.

2. **Class label = folder name.** Only two classes:
   - `pufferfish` — *Tetraodontidae* species
   - `other_fish`  — any other bycatch species

3. **Image formats:** JPEG, PNG, BMP, WebP, TIFF (≥128×128 pixels recommended).

4. **No haul spreadsheet images.** Do not derive labels from the ring-seine haul
   occurrence records. Image labels must come from direct visual identification.

## Recommended sources

| Source | URL | Notes |
|--------|-----|-------|
| iNaturalist Tetraodontidae | https://www.inaturalist.org/taxa/47174-Tetraodontidae/observations | CC-BY, CC0 |
| GBIF StillImage | https://www.gbif.org/occurrence/search?taxon_key=5166&mediaType=StillImage | Check per-record licence |
| FathomNet | https://fathomnet.org | Underwater; good for other_fish |

## Training

After collecting ≥100 images per class per environment:

```bash
# Split raw → train/val/test, then train
python scripts/train_camera_model.py \
    --split-raw data/camera/boat_deck \
    --data-dir data/camera/boat_deck_split \
    --env boat_deck \
    --epochs 20

python scripts/train_camera_model.py \
    --split-raw data/camera/underwater \
    --data-dir data/camera/underwater_split \
    --env underwater \
    --epochs 20
```

Model artifacts are saved to `artifacts/camera/`.
