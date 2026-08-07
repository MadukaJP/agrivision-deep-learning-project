# feature_log.md — AgriVision AI Build Log

## Sprint 1 — Pipeline setup & dataset sourcing
- Scoped model architecture (EfficientNetB3, two-phase transfer learning),
  data sources (Cassava competition dataset + PlantVillage-derivative
  `vipoooool/new-plant-diseases-dataset`), and dropped yam from scope
  (no usable public labeled dataset found).
- Built `config.py`, `data/dataset_loader.py` (unified manifest across
  both sources, stratified train/val/test split, augmentation pipeline).
- Verified Kaggle dataset mount paths via `os.walk` before building the
  manifest — caught and corrected incorrect assumed folder paths for
  both the competition dataset (nested under `/competitions/`) and the
  PlantVillage-derivative dataset (nested under `/datasets/vipoooool/`).

## Sprint 2 — Phase 1 training, first major bug
- Manifest built successfully: 58,348 images across 21 classes (4 crops).
  Confirmed real, expected class imbalance (cassava_mosaic_disease at
  13,158 images vs. cassava_bacterial_blight at 1,087).
- **Bug found**: initial Phase 1 training run stayed stuck near
  random-guess accuracy (~4-5%) for 7 straight epochs. Diagnosed via a
  tiny-subset overfit test (confirmed model/pipeline could learn) and an
  augmentation-ablation test (confirmed the bug was specific to the
  augmentation step). Root cause: `RandomContrast`/`RandomBrightness`
  pushed pixel values outside `[0,1]` without clipping, which then got
  multiplied into an invalid range before EfficientNet's preprocessing —
  feeding the network corrupted input on every training image.
- **Fix**: added explicit `tf.clip_by_value(..., 0.0, 1.0)` after
  augmentation; reduced augmentation intensity; removed cutout
  (reintroducible later if needed).
- Phase 1 completed successfully after the fix: 52.9% val accuracy,
  val_loss 1.552, best at epoch 12/12 (still improving — good sign).

## Sprint 3 — Phase 2 training, two more bugs
- **Bug found**: first Phase 2 attempt crashed — `ReduceLROnPlateau`
  callback is incompatible with a `CosineDecay` learning-rate schedule
  (tries to directly assign a value to a schedule object).
- **Fix**: removed `ReduceLROnPlateau` from the Phase 2 callback list,
  relying on `CosineDecay`'s built-in schedule instead.
- **Bug found**: retry attempt ran without crashing, but val_accuracy
  regressed well below Phase 1's ending point (52.9% down to ~28-35%)
  instead of improving. Isolated via a targeted diagnostic — reloading
  the Phase 1 checkpoint and running `model.evaluate()` before any
  unfreezing confirmed the checkpoint itself loaded correctly (exact
  match: 0.5288). This narrowed the regression to the unfreeze step.
  Root cause: BatchNormalization layers within the newly-unfrozen top 40
  backbone layers were being destabilized by small-batch (16), heavily
  class-weighted training.
- **Fix**: explicitly kept BatchNorm layers frozen within the unfrozen
  range, while leaving Conv/Dense layers trainable.
- Phase 2 completed successfully after the fix: 59.5% val accuracy,
  val_loss 1.316, best at epoch 14/20 (EarlyStopping restored these
  weights after 5 epochs without further improvement).

## Sprint 4 — Evaluation & explainability
- Ran final evaluation on the held-out test set (5,835 images, never
  touched until this point): 58% overall accuracy — closely matching
  validation accuracy, confirming no data leakage.
- Classification report showed strong performance on maize/pepper
  (0.67-0.98 F1), mixed on tomato, and a clear weakness in cassava's
  4-way disease discrimination (0.26-0.58 F1). Confusion matrix
  confirmed nearly all model confusion is contained within the cassava
  class block, not bleeding across crops.
- **Bug found**: initial Grad-CAM implementation produced a heatmap and
  prediction that disagreed with the real model's `model.predict()`
  output on the identical image. Root cause: the Grad-CAM code path fed
  raw `[0,1]`-scaled pixels into the EfficientNetB3 backbone directly,
  skipping the `×255` + `preprocess_input` step the real inference path
  applies — producing an unreliable heatmap from corrupted activations.
- **Fix**: applied identical preprocessing in the Grad-CAM path before
  the backbone call.
- Generated Grad-CAM comparison evidence: a misclassified cassava_mosaic
  image showed peak heatmap activation on background soil (Defense
  Question 1 evidence); a correctly classified case showed primary
  activation on the actual lesion pattern, with a smaller residual
  background hotspot — showing augmentation reduced but didn't fully
  eliminate background-shortcut reliance.
- Ran OOD (out-of-distribution) threshold test on 3 non-plant images:
  2 of 3 were classified with 98-100% confidence into an arbitrary
  disease class (threshold failed to catch them); 1 of 3 was correctly
  flagged as uncertain. Documented as an honest limitation (Defense
  Question 3) rather than presented as a working safeguard.
- Wrote `DEFENSE_ANSWERS.md` with all three answers backed by this real
  evidence.

## Sprint 5 — App build & deployment prep
- Built Streamlit app first (upload → predict → Grad-CAM → PDF report),
  then switched to Flask for Render deployment after ruling out Vercel
  (TensorFlow's install size exceeds Vercel's serverless function
  size limit).
- Built Flask app (`app.py`, HTML templates, PDF report generation via
  `fpdf2`), fixed a path-duplication issue by having `app.py` reference
  the trained model from `models/saved_models/` via a relative path
  rather than keeping a duplicate copy inside `app/`.
- Tested locally end-to-end (upload → prediction → Grad-CAM overlay →
  PDF download) — confirmed working before deployment.
- Updated `requirements.txt` to match the real, tested local environment
  (TensorFlow 2.21.0 / Keras 3.15.1 / NumPy 2.5.1) after discovering the
  originally-planned TensorFlow 2.17.0 pin had no compatible wheel for
  the local Python version.
