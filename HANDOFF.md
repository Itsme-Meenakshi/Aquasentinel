# AquaSentinel — Project Handoff Document

**Date:** October 28, 2026  
**Project:** AquaSentinel (Pufferfish Occurrence Monitoring & Digital Twin for Central Kerala Coast)  
**Author/Session:** Antigravity to VS Code Handoff  
**Status:** All 51 Unit Tests Passing (`pytest`); Ready for Local VS Code Development

---

## 1. Project Purpose & Current State

AquaSentinel is a BTech major project (Group 15, MITS Kochi) that monitors and forecasts pufferfish (*Tetraodontidae*) bycatch risk along the central Kerala coast (Kochi region). The system combines:
1. **Haul Occurrence Predictive Models:** Chronologically-validated machine learning models evaluating seasonal, depth, and sea-surface temperature (SST) factors on historical CMFRI ring-seine hauls.
2. **Digital Twin:** A stateful, versioned representation of coastal observations, out-of-time scoring, refusal logic ("insufficient data"), and human-reviewed updates.
3. **Camera Recognition System (Page 6):** A PyTorch MobileNetV3-Small CNN classifier that detects pufferfish from boat-deck or underwater camera images and video feeds, designed with mandatory human-in-the-loop review before any detection can update haul records.

### Current State
* **Tabular Models & Digital Twin:** Fully operational. All baseline and logistic models run and pass all checks.
* **Camera Recognition (Boat Deck Model):** Model is trained (`artifacts/camera/boat_deck_mobilenet_v3_small/model.pth`), integrated into Streamlit (`app/pages/6_Camera_Recognition.py`), and operational for single-image and video inference.
* **Camera Recognition (Underwater Model):** Stub model in place (`artifacts/camera/underwater_mobilenet_v3_small/`), awaiting real underwater training images.
* **Test Suite:** **51 of 51 tests pass** across tabular features, models, SST processing, Digital Twin state, and computer vision / inference pipelines.
* **Latest Investigation:** Completed a root-cause investigation into a false positive (91.8% pufferfish prediction on a non-pufferfish boat-deck image). The investigation proved that the model, softmax math, and class indexing are technically correct, but the model suffers from domain bias (all `other_fish` training data came from studio fish-market photos, whereas `pufferfish` came from outdoor boat-deck photos).

---

## 2. Technology Stack & Architecture

* **Language:** Python 3.10+ (tested on Python 3.11.9, Windows 64-bit)
* **Web UI:** Streamlit (multi-page application in `app/`)
* **ML / Tabular:** Scikit-learn, Pandas (2.0+ compatible), NumPy, SciPy
* **Computer Vision & Deep Learning:** PyTorch (`torch`, `torchvision`), Pillow, OpenCV (`opencv-python-headless`)
* **Data & Geo:** OpenPyXL, Requests, PyYAML
* **Testing:** Pytest

### Architectural Flow

```
Raw Data Sources:
  ├── CMFRI Ring-seine hauls (Excel/CSV)
  ├── NOAA OISST Sea Surface Temperature
  └── Boat Deck & Underwater Images (data/camera/)
             │
             ▼
Core Library (`src/pufferfish/`):
  ├── data/           → Loaders and schema checks
  ├── features/       → Cyclical day-of-year & environmental features
  ├── environmental/  → SST coordinate matching & interpolation
  ├── models/         → Estimators, chronological evaluation, reporting
  ├── twin/           → Digital Twin versioning, state tracking, review
  └── camera/         → MobileNetV3 CNN, dataloaders, inference, video tracking
             │
             ▼
Applications & Deliverables:
  ├── Streamlit App (`app/main.py`, Pages 1-6)
  ├── Pipeline Runner (`scripts/run_pipeline.py`)
  ├── Vision Pipeline (`scripts/train_camera_model.py`, `prepare_other_fish_dataset.py`)
  └── Artifacts & Reports (`artifacts/`, `reports/`)
```

---

## 3. Important Folders & Files

```
Aquasentinel/
├── app/
│   ├── main.py                     # Streamlit app entrypoint
│   ├── _common.py                  # Shared UI components and styles
│   └── pages/
│       ├── 1_Data_Explorer.py      # Spatial & temporal haul data exploration
│       ├── 2_Model_Results.py      # Chronological validation metrics
│       ├── 3_Scenario.py           # What-if scenario testing & risk assessment
│       ├── 4_Report_Haul.py        # Human review interface for pending hauls
│       ├── 5_Twin_History.py       # Versioned state logs & audit trail
│       └── 6_Camera_Recognition.py # CNN image & video classifier with human review
├── artifacts/
│   ├── camera/
│   │   ├── boat_deck_mobilenet_v3_small/  # Trained boat-deck model weights & metadata
│   │   └── underwater_mobilenet_v3_small/ # Stub underwater model
│   ├── models/                     # Serialized tabular models (joblib/pickle)
│   └── twin/                       # Serialized Digital Twin state
├── config/
│   ├── config.yaml                 # Pipeline and SST configuration
│   └── features.yaml               # Feature set definitions
├── data/
│   ├── camera/                     # Image datasets (boat_deck, boat_deck_split, underwater)
│   ├── interim/                    # Cleaned intermediate tabular data
│   ├── processed/                  # Final training data with features
│   └── raw/                        # Original haul Excel sheets
├── docs/                           # Methodology, data dictionary, system architecture
├── reports/                        # Model comparison CSVs, figures, and plots
├── scripts/
│   ├── run_pipeline.py             # Main tabular & twin pipeline execution
│   ├── prepare_other_fish_dataset.py # Scans images.cv download and splits classes
│   ├── pretrain_check.py           # Pre-training dataset audit script (33 checks)
│   ├── train_camera_model.py       # MobileNetV3 PyTorch training script
│   └── _fix_manifest.py            # Manifest cleaner for deleted/moved images
├── src/pufferfish/
│   ├── camera/                     # Vision module: dataset.py, model.py, inference.py, video.py
│   ├── data/, environmental/, features/, geo/, models/, twin/
├── tests/                          # 51 unit tests (conftest.py, test_vision.py, etc.)
├── requirements.txt                # Python package dependencies
└── README.md                       # Project overview and run instructions
```

---

## 4. Work Completed in This Session

1. **Bug Fixes for Pandas 2.0+ Compatibility:**
   * Updated deprecated `dt.dayofyear` to `dt.day_of_year` in `src/pufferfish/features/build.py`, `src/pufferfish/twin/state.py`, and `tests/conftest.py`.
2. **Pytest Test Suite Verification:**
   * Executed full test suite with `.venv/Scripts/python -m pytest`: **51 passed in 21.96s** (including 29 computer vision unit tests).
3. **App Compilation & Validation:**
   * Verified syntax and compilation across `app/main.py` and all 6 Streamlit pages with zero syntax errors.
4. **False-Positive Investigation (91.8% Pufferfish / 8.2% Other Fish):**
   * Identified the exact tested image on disk: `data/camera/boat_deck/pufferfish/imgi_70_medium.jpg` ($225 \times 500$, SHA-256: `74017e67...`).
   * Confirmed the model checkpoint matches the latest training run (Epoch 20).
   * Verified class indexing (`other_fish: 0`, `pufferfish: 1`) is correct across training, inference, and UI.
   * Proved that the false positive was caused by **dataset domain bias / Clever Hans effect**: `other_fish` training data had only cool studio backgrounds ($B>R$), while `pufferfish` had warm outdoor boat-deck backgrounds ($R>B$).
   * Created a comprehensive investigation report artifact: `cnn_false_positive_investigation_report.md`.

---

## 5. Outstanding Bugs, Incomplete Work & Limitations

1. **Domain Bias in `boat_deck` Model:**
   * *Status:* Identified root cause; not yet retrained (per non-destructive instructions).
   * *Issue:* The model misclassifies non-pufferfish photographed on boat decks because the negative training class (`other_fish`) contains no boat-deck imagery.
2. **Duplicate Images in Raw Pufferfish Folder:**
   * *Status:* Raw folder `data/camera/boat_deck/pufferfish` has duplicated images under different names (e.g., `imgi_100_medium.jpg` == `imgi_45_medium.jpg`).
3. **Underwater Model (`underwater_mobilenet_v3_small`):**
   * *Status:* Untrained stub. No underwater training images are currently in `data/camera/underwater/`.
4. **Optional Features in Model Comparison:**
   * Models requiring SST features (`logistic_season_sst`, `logistic_season_depth_sst`, `forest_all_prehaul`) are skipped if SST grids have not been pulled from NOAA OISST.

---

## 6. How to Set Up & Run in VS Code Tomorrow

### Prerequisites
* Windows 10/11 with VS Code installed.
* Python 3.11 (virtual environment already exists at `.venv`).

### Step-by-Step Commands (PowerShell inside VS Code Terminal)

1. **Open Project Folder:**
   * Launch VS Code: `code c:\Users\nouri\OneDrive\Desktop\Aquasentinel`
   * Or via File -> Open Folder -> `C:\Users\nouri\OneDrive\Desktop\Aquasentinel`.

2. **Select Virtual Environment Interpreter:**
   * In VS Code, press `Ctrl+Shift+P` -> `Python: Select Interpreter`.
   * Choose: `.\.venv\Scripts\python.exe`.

3. **Install / Verify Dependencies:**
   ```powershell
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

4. **Run the Test Suite:**
   ```powershell
   .\.venv\Scripts\python -m pytest
   ```
   *(Expected output: `51 passed`)*

5. **Launch the Streamlit Web Application:**
   ```powershell
   .\.venv\Scripts\python -m streamlit run app/main.py
   ```
   * Open browser at `http://localhost:8501`.
   * Navigate to `6_Camera_Recognition` to test the vision module.

6. **Optional: Run the Tabular Pipeline:**
   ```powershell
   .\.venv\Scripts\python scripts/run_pipeline.py --skip-sst
   ```

---

## 7. Environment Variables

* **No external secrets or API keys are required.**
* AquaSentinel runs entirely offline and on local CPU.
* Optional NOAA OISST downloads connect via anonymous public HTTP requests defined in `config/config.yaml`.

---

## 8. Exact Next Steps to Continue Tomorrow (Priority Order)

1. **Priority 1: Add In-Domain `other_fish` Negative Samples**
   * Source 50–100 non-pufferfish images taken on boat decks, in catch baskets, or in fishermen's hands (e.g. from iNaturalist observations of common bycatch fish).
   * Place them into `data/camera/boat_deck/other_fish/`.
2. **Priority 2: Clean Raw Duplicates & Re-split Dataset**
   * Deduplicate `data/camera/boat_deck/pufferfish/`.
   * Re-run split generation:
     ```powershell
     .\.venv\Scripts\python scripts/train_camera_model.py --split-raw data/camera/boat_deck --data-dir data/camera/boat_deck_split --env boat_deck --epochs 20
     ```
3. **Priority 3: Evaluate Fine-Tuning with Background Invariance**
   * Verify on `data/camera/boat_deck/pufferfish/imgi_70_medium.jpg` that the false positive resolves after in-domain negative data is added.
4. **Priority 4: Commit Workspace Changes to Git**
   * Review `git status` and commit working changes to `main` branch.

---

## 9. Critical Design Decisions (Do Not Change)

* **Separation of Environments:** `boat_deck` and `underwater` models must remain completely separate directories and models. Never mix underwater and deck imagery into the same training set.
* **Human-in-the-Loop Safeguard:** Camera model predictions must **never** automatically insert or modify occurrence records in the Digital Twin. Detections must be reviewed and accepted by a human in `app/pages/4_Report_Haul.py` or `6_Camera_Recognition.py`.
* **Chronological Splits for Haul Models:** Tabular models must only be evaluated on future chronological splits, never random k-fold cross-validation, to prevent temporal leakage.
* **CPU Compatibility:** All PyTorch code and inference pipelines are intentionally restricted to CPU execution to guarantee portability on field laptops.

