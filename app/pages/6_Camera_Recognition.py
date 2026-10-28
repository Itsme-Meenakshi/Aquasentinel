"""Camera Recognition — AquaSentinel Dashboard Page 6.

Supports:
- Single image upload → binary classification (pufferfish / other_fish)
- Video upload → frame sampling, event detection, human review
- Model status display (trained vs stub)
- CSV export of predictions and candidate events
- Human review controls (accept / reject / uncertain) before any haul integration
"""
from __future__ import annotations
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pufferfish.camera.inference import PufferfishClassifier
from pufferfish.camera.video import (
    classify_video, group_candidate_events,
    export_predictions_csv, export_events_csv, get_video_info,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Camera Recognition · AquaSentinel",
    page_icon="🐡",
    layout="wide",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* gradient header */
.cam-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.cam-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.cam-header p  { font-size: 0.95rem; opacity: 0.78; margin: 0.4rem 0 0; }

/* model status cards */
.status-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
}
.status-trained   { background: #0d2e1a; border-color: #22c55e; color: #86efac; }
.status-stub      { background: #2a1a07; border-color: #f97316; color: #fdba74; }

/* prediction result cards */
.pred-positive {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    color: white;
}
.pred-negative {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    color: white;
}
.pred-label { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.5px; }
.pred-conf  { font-size: 1rem; opacity: 0.8; margin-top: 0.25rem; }

/* event cards */
.event-card {
    background: #1a1f2e;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.event-title { font-weight: 600; font-size: 1rem; color: #e2e8f0; }
.event-meta  { font-size: 0.82rem; color: #94a3b8; margin-top: 0.25rem; }

/* warning banner */
.human-review-note {
    background: #1c1400;
    border: 1px solid #ca8a04;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #fde68a;
    margin-bottom: 1rem;
}

/* confidence bar wrapper */
.conf-bar { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
ARTIFACT_DIR   = ROOT / "artifacts" / "camera"
ENVIRONMENTS   = {"Boat / Deck": "boat_deck", "Underwater": "underwater"}
ARCH           = "mobilenet_v3_small"
VIDEO_EXTS     = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
IMAGE_EXTS     = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cam-header">
    <h1>🐡 Camera Recognition</h1>
    <p>Upload images or videos to detect pufferfish. All predictions require human review before integration with haul records.</p>
</div>
""", unsafe_allow_html=True)

# ── human review notice ───────────────────────────────────────────────────────
st.markdown("""
<div class="human-review-note">
    ⚠️ <strong>Human Review Required:</strong> Camera predictions are <em>never</em> automatically added to haul records.
    Every detection must be reviewed and confirmed before any occurrence data is updated.
</div>
""", unsafe_allow_html=True)

# ── sidebar — model selection ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    env_label = st.selectbox("Environment", list(ENVIRONMENTS.keys()))
    env_key   = ENVIRONMENTS[env_label]
    model_dir = ARTIFACT_DIR / f"{env_key}_{ARCH}"

    # Load (or cache) classifier
    @st.cache_resource(show_spinner="Loading classifier…")
    def get_classifier(model_dir_str: str):
        try:
            return PufferfishClassifier(Path(model_dir_str)), None
        except Exception as exc:
            return None, str(exc)

    clf, clf_err = get_classifier(str(model_dir))

    # Model status card
    if clf_err:
        st.markdown(f'<div class="status-card status-stub">⚠️ Classifier not loaded<br><small>{clf_err}</small></div>',
                    unsafe_allow_html=True)
    elif clf:
        s = clf.status()
        if s["trained"]:
            ev = s.get("evaluation") or {}
            acc = ev.get("accuracy", "—")
            f1  = ev.get("macro_f1", "—")
            st.markdown(
                f'<div class="status-card status-trained">✅ <strong>Model trained</strong><br>'
                f'<small>Arch: {s["architecture"]} · Env: {s["environment"]}<br>'
                f'Accuracy: {acc} · F1: {f1}</small></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="status-card status-stub">⚠️ <strong>Untrained stub</strong><br>'
                f'<small>Arch: {s["architecture"]} · Env: {s["environment"]}<br>'
                f'Run <code>scripts/train_camera_model.py</code> to train.</small></div>',
                unsafe_allow_html=True)
        if s.get("notes"):
            st.caption(s["notes"])

    st.divider()
    st.subheader("Video Settings")
    sample_interval = st.slider("Sample interval (s)", 0.5, 5.0, 1.0, 0.5)
    gap_tolerance   = st.slider("Event gap tolerance (s)", 1.0, 10.0, 3.0, 0.5)
    min_pos_frames  = st.slider("Min positive frames per event", 1, 10, 2, 1)
    max_frames      = st.slider("Max frames to process", 50, 1000, 300, 50)

    st.divider()
    st.markdown("""
**About models**
- `boat_deck` — surface / deck photos
- `underwater` — submerged camera footage
- Keep separate; do not mix environments.

**Training data**
Place images under `data/camera/<env>/train/`, `val/`, `test/`
with `pufferfish/` and `other_fish/` subfolders.
""")

# ── main tabs ─────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_help = st.tabs(["📷 Image", "🎥 Video", "ℹ️ Help & Data Setup"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.subheader("Upload Images")
    uploaded_images = st.file_uploader(
        "Select one or more fish images",
        type=[e.lstrip(".") for e in IMAGE_EXTS],
        accept_multiple_files=True,
        key="img_uploader",
    )

    if not clf:
        st.error("Classifier unavailable — check the sidebar for details.")
    elif uploaded_images:
        if not clf.is_trained:
            st.warning("⚠️ This is an **untrained stub model** — predictions are meaningless. "
                       "Train the model first using `scripts/train_camera_model.py`.")

        st.markdown(f"**{len(uploaded_images)} image(s) uploaded** — classifying with `{env_key}` model…")
        results_data = []

        n_cols = min(3, len(uploaded_images))
        cols = st.columns(n_cols)

        for i, uf in enumerate(uploaded_images):
            col = cols[i % n_cols]
            with col:
                img_bytes = uf.read()
                st.image(img_bytes, caption=uf.name, use_container_width=True)

                # Run inference via temp file
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uf.name).suffix, delete=False
                ) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    result = clf.predict_image(tmp_path)
                    cls_name   = result["class_name"]
                    confidence = result["confidence"]
                    probs      = result["probabilities"]
                    is_puffer  = result["is_pufferfish"]

                    card_cls  = "pred-positive" if is_puffer else "pred-negative"
                    emoji     = "🐡" if is_puffer else "🐟"
                    label_txt = "Pufferfish" if is_puffer else "Other fish"

                    st.markdown(
                        f'<div class="{card_cls}">'
                        f'<div class="pred-label">{emoji} {label_txt}</div>'
                        f'<div class="pred-conf">Confidence: {confidence:.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Mini probability bars
                    for cname, prob in probs.items():
                        st.progress(prob, text=f"{cname}: {prob:.1%}")

                    results_data.append({
                        "filename": uf.name,
                        "class_name": cls_name,
                        "confidence": round(confidence, 4),
                        "pufferfish_prob": round(probs.get("pufferfish", 0.0), 4),
                        "other_fish_prob": round(probs.get("other_fish", 0.0), 4),
                        "environment": env_key,
                        "model_trained": clf.is_trained,
                        "review_status": "unreviewed",
                    })
                except Exception as e:
                    st.error(f"Inference error: {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

        # Export results CSV
        if results_data:
            st.divider()
            import pandas as pd
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name=f"camera_predictions_{env_key}_{int(time.time())}.csv",
                mime="text/csv",
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader(
        "Select a video file",
        type=[e.lstrip(".") for e in VIDEO_EXTS],
        key="vid_uploader",
    )

    if not clf:
        st.error("Classifier unavailable — check the sidebar for details.")
    elif uploaded_video:
        if not clf.is_trained:
            st.warning("⚠️ **Untrained stub** — video predictions will be random. "
                       "Train the model first using `scripts/train_camera_model.py`.")

        # Save video to temp file
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_video.name).suffix, delete=False
        ) as tmp_vid:
            tmp_vid.write(uploaded_video.read())
            vid_path = Path(tmp_vid.name)

        try:
            info = get_video_info(vid_path)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", f"{info.get('duration_sec', '?'):.1f}s")
            col2.metric("FPS", f"{info.get('fps', '?'):.1f}")
            col3.metric("Resolution", f"{info.get('width','?')}×{info.get('height','?')}")
            col4.metric("Total frames", info.get('frame_count', '?'))

            frames_to_scan = min(
                int(info.get("duration_sec", 60) / sample_interval),
                max_frames
            )
            st.info(f"Will sample ~{frames_to_scan} frames at {sample_interval}s intervals.")

            if st.button("▶️ Run Inference", type="primary", key="run_video"):
                with st.spinner(f"Classifying {frames_to_scan} frames…"):
                    tmp_frame_dir = Path(tempfile.mkdtemp(prefix="aqua_frames_"))
                    frame_preds = classify_video(
                        classifier=clf,
                        video_path=vid_path,
                        sample_interval_sec=sample_interval,
                        max_frames=max_frames,
                        temp_dir=tmp_frame_dir,
                    )

                n_positive = sum(1 for p in frame_preds if p.is_pufferfish)
                st.success(
                    f"Processed {len(frame_preds)} frames — "
                    f"**{n_positive} positive** ({n_positive/max(1,len(frame_preds)):.1%})"
                )

                # Candidate events
                events = group_candidate_events(
                    frame_preds,
                    gap_tolerance_sec=gap_tolerance,
                    min_positive_frames=min_pos_frames,
                )

                st.subheader(f"Candidate Events: {len(events)}")

                # Session state for review
                if "event_reviews" not in st.session_state:
                    st.session_state["event_reviews"] = {}

                for ev in events:
                    with st.container():
                        st.markdown(
                            f'<div class="event-card">'
                            f'<div class="event-title">Event #{ev.event_id}</div>'
                            f'<div class="event-meta">'
                            f'⏱ {ev.start_time_sec:.1f}s – {ev.end_time_sec:.1f}s &nbsp;|&nbsp; '
                            f'{ev.n_positive_frames} positive frames &nbsp;|&nbsp; '
                            f'max conf: {ev.max_confidence:.1%} &nbsp;|&nbsp; '
                            f'mean conf: {ev.mean_confidence:.1%}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                        review_key = f"event_{ev.event_id}_review"
                        notes_key  = f"event_{ev.event_id}_notes"
                        current_status = st.session_state["event_reviews"].get(
                            review_key, "unreviewed"
                        )
                        with col_a:
                            notes = st.text_input("Notes", key=notes_key,
                                                  placeholder="Location, depth, gear…")
                        with col_b:
                            if st.button("✅ Accept", key=f"accept_{ev.event_id}"):
                                st.session_state["event_reviews"][review_key] = "accepted"
                                ev.review_status = "accepted"
                                ev.reviewer_notes = notes
                        with col_c:
                            if st.button("❌ Reject", key=f"reject_{ev.event_id}"):
                                st.session_state["event_reviews"][review_key] = "rejected"
                                ev.review_status = "rejected"
                                ev.reviewer_notes = notes
                        with col_d:
                            if st.button("❓ Uncertain", key=f"uncertain_{ev.event_id}"):
                                st.session_state["event_reviews"][review_key] = "uncertain"
                                ev.review_status = "uncertain"
                                ev.reviewer_notes = notes

                        # Update review from session state
                        stored = st.session_state["event_reviews"].get(review_key)
                        if stored:
                            ev.review_status = stored
                            colour = {"accepted": "🟢", "rejected": "🔴",
                                     "uncertain": "🟡"}.get(stored, "⚪")
                            st.caption(f"{colour} Status: **{stored}**")

                # Export section
                st.divider()
                col_dl1, col_dl2 = st.columns(2)

                # Frame-level CSV
                with col_dl1:
                    frame_buf = io.StringIO()
                    import csv as _csv
                    from dataclasses import asdict
                    writer = _csv.DictWriter(frame_buf, fieldnames=[
                        "frame_number", "timestamp_sec", "class_name", "confidence", "is_pufferfish"
                    ])
                    writer.writeheader()
                    for p in frame_preds:
                        writer.writerow(asdict(p))
                    st.download_button(
                        "⬇️ Download frame predictions CSV",
                        data=frame_buf.getvalue().encode(),
                        file_name=f"frame_predictions_{env_key}_{int(time.time())}.csv",
                        mime="text/csv",
                        key="dl_frames",
                    )

                # Events CSV
                with col_dl2:
                    events_buf = io.StringIO()
                    writer2 = _csv.DictWriter(events_buf, fieldnames=[
                        "event_id", "start_frame", "end_frame", "start_time_sec",
                        "end_time_sec", "n_positive_frames", "max_confidence",
                        "mean_confidence", "review_status", "reviewer_notes"
                    ])
                    writer2.writeheader()
                    for e in events:
                        writer2.writerow(asdict(e))
                    st.download_button(
                        "⬇️ Download events CSV",
                        data=events_buf.getvalue().encode(),
                        file_name=f"candidate_events_{env_key}_{int(time.time())}.csv",
                        mime="text/csv",
                        key="dl_events",
                    )

                # Frame-level chart
                if frame_preds:
                    import pandas as pd
                    df_frames = pd.DataFrame([
                        {"time_s": p.timestamp_sec, "pufferfish_prob": 
                         p.confidence if p.is_pufferfish else 1.0 - p.confidence}
                        for p in frame_preds
                    ])
                    st.subheader("Pufferfish confidence over time")
                    st.line_chart(df_frames.set_index("time_s")["pufferfish_prob"])

        except Exception as e:
            st.error(f"Video processing error: {e}")
        finally:
            vid_path.unlink(missing_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — HELP
# ════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.subheader("Data Setup Guide")
    st.markdown("""
### 1 — Directory structure

Create one image directory per environment:

```
data/
  camera/
    boat_deck/
      pufferfish/   ← pufferfish photos from the boat/deck
      other_fish/   ← other fish from the boat/deck
    underwater/
      pufferfish/   ← pufferfish in underwater footage
      other_fish/   ← other fish in underwater footage
```

**Important:** Do NOT mix environments. Each environment trains a separate model.

---

### 2 — Image sources

| Source | Format | Recommended for |
|--------|--------|-----------------|
| [iNaturalist — Tetraodontidae](https://www.inaturalist.org/taxa/47174-Tetraodontidae) | JPEG | pufferfish (both environments) |
| [GBIF Multimedia](https://www.gbif.org/occurrence/search?taxon_key=5166&mediaType=StillImage) | JPEG | pufferfish |
| [FathomNet](https://fathomnet.org) | JPEG | underwater other_fish |
| Your haul photos / GoPro footage | MP4/JPEG | both |

---

### 3 — Prepare splits and train

```bash
# Step 1: Split raw images (auto 70/15/15 split)
python scripts/train_camera_model.py \\
    --split-raw data/camera/boat_deck \\
    --data-dir data/camera/boat_deck_split \\
    --env boat_deck

# Step 2: Train the model
python scripts/train_camera_model.py \\
    --data-dir data/camera/boat_deck_split \\
    --env boat_deck \\
    --arch mobilenet_v3_small \\
    --epochs 20
```

Repeat for `underwater`.

---

### 4 — Human review workflow

1. Upload image or video in this page
2. Review each candidate detection
3. **Accept** only detections with sufficient confidence + visual confirmation
4. Export reviewed CSV for recordkeeping
5. Only accepted, human-reviewed detections should be cross-referenced with haul records

**Camera predictions NEVER automatically update occurrence records or the predictive model.**
""")

    st.divider()
    st.subheader("Model Architecture")
    st.markdown("""
- **Base:** MobileNetV3-Small (ImageNet pretrained)
- **Fine-tuned:** Final classifier layer only (binary: pufferfish / other_fish)
- **Input:** 224×224 RGB, ImageNet normalised
- **Device:** CPU-compatible (no GPU required)
- **Per-environment:** `boat_deck` and `underwater` models are independent
""")
