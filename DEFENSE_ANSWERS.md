# DEFENSE_ANSWERS.md — AgriVision AI

## Question 1: Grad-CAM on a leaf against a noisy background — did the model
focus on the lesion or the background? What augmentation change fixed it
(or didn't)?

**Finding:** Tested on a true `cassava_mosaic_disease` test-set image with a
messy soil/shadow background. Two cases were compared:

- **Misclassified case**: the model predicted `tomato_yellow_leaf_curl_virus`
  (a different crop entirely). The Grad-CAM heatmap showed its strongest,
  most concentrated activation on a background hotspot in the image corner
  (soil/lighting artifact), not on the leaf. The model was reading the
  background, not the lesion, and got the prediction wrong.
- **Correctly classified case** (same image characteristics, different
  sample): the model correctly predicted `cassava_mosaic_disease`. The
  heatmap's primary, strongest activation sat directly on the leaf's mottled
  discoloration pattern — the actual visual signature of the disease.
  However, a smaller secondary hotspot remained on the same background
  corner region seen in the misclassified case.

**What this shows:** the augmentation pipeline used (random flip, rotation,
zoom, contrast/brightness jitter within `[0,1]` with explicit clipping,
translation) measurably reduced background-shortcut reliance — the model's
*primary* signal correctly shifted to lesion content in the successful case
— but did not eliminate it completely, evidenced by the residual secondary
background activation even on the correct prediction.

**Honest limitation / future work:** a stronger fix would use a tighter
random-crop that forces more leaf-fill per training image, or an explicit
leaf-segmentation preprocessing step to mask out background before the
model sees it. Neither was implemented due to project scope/time.

**Bug found and fixed along the way:** the initial Grad-CAM implementation
fed raw `[0,1]`-scaled pixels directly into the EfficientNetB3 backbone
without applying the same `×255` + `preprocess_input` step the real model
pipeline uses. This produced a visibly different (wrong) prediction than
`model.predict()` on the identical image, and therefore an unreliable
heatmap. Confirmed via side-by-side comparison of `model.predict()` output
vs. the Grad-CAM path's internal prediction, then fixed by applying
identical preprocessing before the backbone call in both paths.

---

## Question 2: At what exact epoch did Phase 2 overfitting begin, and what
resolved it?

**Finding:** Phase 2 fine-tuning ran with a `CosineDecay` learning rate
schedule (initial LR `1e-5`, decayed over `steps_per_epoch × 20` epochs) and
`EarlyStopping(patience=5, restore_best_weights=True)` monitoring `val_loss`.

- Best `val_loss` (1.3160) occurred at **epoch 14** (`val_accuracy` 59.5%).
- Training continued to roughly epoch 18-19 without further `val_loss`
  improvement, at which point `EarlyStopping` triggered and restored the
  epoch-14 weights.

**What resolved it:** not a manual learning-rate drop — this run
deliberately did NOT use `ReduceLROnPlateau` (an earlier attempt combining
it with `CosineDecay` crashed, since `ReduceLROnPlateau` tries to directly
assign a new value to what is already a schedule object, which is invalid).
Instead, two things protected the model from the overfitting that began
after epoch 14: (1) the already-smoothly-decaying `CosineDecay` schedule,
which had reduced the LR to roughly `2×10⁻⁶` by epoch 14, and (2)
`EarlyStopping` with `restore_best_weights=True`, which discarded the
overfit epochs 15-19 and kept the epoch-14 weights as the final saved model.

**A real bug found and fixed during this phase:** an earlier Phase 2 attempt
(loading the Phase 1 checkpoint and unfreezing the top 40 backbone layers)
caused `val_accuracy` to *regress* below Phase 1's ending accuracy (52.9%
down to ~28-35%) instead of improving on it. Diagnosed via a targeted test:
reloading the Phase 1 checkpoint and calling `model.evaluate()` on the
validation set *before* any unfreezing confirmed the checkpoint loaded
correctly (reproduced 52.9% exactly). This isolated the regression to the
unfreeze step itself — specifically, BatchNormalization layers within the
newly-unfrozen top 40 layers being disrupted by small-batch (16),
heavily class-weighted training. Fix: explicitly kept all BatchNorm layers
within the unfrozen range frozen (`trainable = False`), while leaving
Conv/Dense layers in that range trainable. This resolved the regression —
the corrected run started at ~51% val_accuracy in epoch 1 (matching Phase 1's
end point) instead of collapsing.

---

## Question 3: How was OOD (out-of-distribution) rejection implemented, and
does it work?

**Implementation:** a softmax max-confidence threshold
(`OOD_CONFIDENCE_THRESHOLD = 0.60`) — if the model's top predicted class
probability falls below this threshold, the app reports "uncertain /
not a recognized crop-disease" instead of forcing a diagnosis.

**Finding — honestly, it does not work reliably.** Tested against three
non-plant photos (none were leaves):

| Test image | Predicted class | Confidence | Result |
|---|---|---|---|
| Non-plant photo 1 | maize_common_rust | 1.00 | FAILED — overconfident |
| Non-plant photo 2 | maize_gray_leaf_spot | 0.98 | FAILED — overconfident |
| Non-plant photo 3 | cassava_bacterial_blight | 0.49 | PASSED — correctly flagged uncertain |

2 of 3 non-plant images were classified with near-total (98-100%)
confidence into an arbitrary disease class — the threshold failed to catch
them. Only 1 of 3 produced a low enough confidence to be correctly rejected.

**Why this happens:** softmax always outputs a probability distribution
that sums to 1 across the model's known classes, regardless of input. The
network was never trained on anything outside its 21 classes and has no
built-in concept of "none of the above" — given an unfamiliar image, it
simply reports whichever of its 21 known answers activated most strongly,
often with high confidence, because from the model's internal perspective
one of those 21 has to be the answer.

**Honest limitation / future work:** a softmax confidence threshold is a
weak, unreliable OOD mechanism on its own. Two more robust alternatives,
not implemented here due to project scope: (1) train an explicit extra
"background / not-a-leaf" class using a sample of random non-plant images,
giving the model something concrete to compare against; or (2) use a
distance-based OOD method — comparing the image's feature-space embedding
against the training data's distribution and flagging anything too far
outside it, independent of softmax confidence entirely.
