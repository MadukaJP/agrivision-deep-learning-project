# AgriVision AI — Build & Run Guide

## Datasets to add on Kaggle

1. `cassava-leaf-disease-classification` (competition — join rules first)
2. `vipoooool/new-plant-diseases-dataset` — covers maize, tomato, pepper.
   Note: this dataset's `train/` folder contains augmentation-generated
   variants of the same base images used in `valid/`. Pooling both (as
   `dataset_loader.py` does) and re-splitting ourselves is fine since the
   two folders contain different underlying source photos, not exact
   duplicates — but if you ever see suspiciously perfect validation
   accuracy, this pooling is the first thing to double check.

## Run order (Kaggle notebook)

1. **Verify paths first.** Before anything else, run:
   ```python
   from data.dataset_loader import verify_paths
   verify_paths()
   ```
   Fix any `[MISSING]` paths in `config.py` before proceeding — folder
   names in PlantVillage mirrors are inconsistent, don't assume the
   defaults in `config.py` are correct for your specific dataset add.

2. **Build the manifest:**
   ```python
   from data.dataset_loader import build_manifest, split_manifest
   df, class_to_idx = build_manifest()
   train_df, val_df, test_df = split_manifest(df)
   ```
   Expect ~20 unified classes across cassava, maize, tomato, and pepper.
   Check the printed `value_counts()` — if any class has under ~100
   images, note it, this affects how much you should trust that class's
   individual precision/recall later.

3. **Train:**
   ```
   python models/train.py
   ```
   or import and call `main()` directly in a notebook cell (recommended
   on Kaggle so you can watch progress and restart from Phase 2 if the
   session times out after Phase 1).

   **What to expect:**
   - Phase 1 (frozen backbone): fast per epoch, accuracy should climb
     steadily. If val_accuracy plateaus far below train_accuracy, that's
     overfitting even in Phase 1 — check class imbalance handling.
   - Phase 2 (fine-tuning): much slower per epoch. Watch `phase2_history.csv`
     — the epoch where `val_loss` stops decreasing and starts rising is
     your literal answer to Defense Question 2. Don't just eyeball the
     printed plot — open the CSV and find the exact epoch number.
   - Total training time: budget 2-4 hours depending on dataset size and
     GPU (T4 vs P100). Don't start this with only 1 hour of session time left.

4. **Evaluate:**
   ```
   python models/evaluate.py
   ```
   This runs on the held-out test set only — never touched until now.
   Also manually run:
   - `test_gradcam_on_sample()` on a cassava image with a messy/soil
     background (Defense Question 1 evidence).
   - `test_ood_rejection()` on a photo of literally anything that isn't
     a leaf — a shoe, a wall, your hand (Defense Question 3 evidence).

5. **Download outputs** from Kaggle's Output tab before your session
   ends: `agrivision_final.keras`, `class_index.json`, everything in
   `models/plots/`, and `models/logs/*.csv`.

## Deployment (Render — Docker)

The app deploys on Render as a **Docker** service. Docker pins Python 3.12
inside the image, which guarantees `tensorflow==2.21.0` (cp312 Linux wheel)
installs — this avoids Render's native `runtime.txt` fallback issue where
an unsupported version silently falls back to Python 3.14 (no TF wheel).

### Option A — Blueprint (recommended, one click)

1. Push this repo to GitHub (model weights live in `models/saved_models/`).
2. Render dashboard → **New + → Blueprint** → connect `MadukaJP/agrivision-deep-learning-project`.
3. Render reads `render.yaml` and creates the `agrivision` web service
   (`runtime: docker`, `dockerfilePath: ./Dockerfile`). Deploy starts automatically.

### Option B — Manual web service

1. Render dashboard → **New + → Web Service** → connect the GitHub repo.
2. Render detects the `Dockerfile` — **Runtime = Docker**.
3. Service will build the image and run the `CMD`:
   `gunicorn --chdir /app/app --bind 0.0.0.0:$PORT --workers 1 --threads 4 app:app`

### Notes

- **RAM:** Free tier is 512 MB. TensorFlow uses ~500 MB just loading the
  model, so expect slow starts / possible OOM on free. If the container is
  killed on startup, upgrade to **Starter** (Settings → Instance Type).
- **Storage:** `app/static/uploads/` is ephemeral — wiped on every restart.
  Uploaded images/PDFs are demo-only.
- **Local test:**
  ```
  docker build -t agrivision .
  docker run -p 8000:8000 agrivision
  ```
  Then open http://localhost:8000.

## What NOT to skip

The three defense questions are graded on whether you can produce real
evidence, not a plausible-sounding answer. Specifically save:
- `phase1_curves.png` and `phase2_curves.png` (loss/accuracy plots)
- `phase2_history.csv` (to cite the exact overfitting epoch)
- `gradcam_noisy_bg.png` (or similar name) showing the messy-background test
- The printed OOD test output (confidence score on a non-plant image)

These four artifacts are what turn "I built a CNN" into "I can defend
what happened when I trained it" — which is the actual point of the
rubric's defense section.
