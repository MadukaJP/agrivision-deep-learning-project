"""
models/evaluate.py — AgriVision AI

Run this ONLY after train.py has produced agrivision_final.keras and
class_index.json. This script:
  1. Evaluates on the held-out test set (never seen during training/val).
  2. Generates a confusion matrix + classification report.
  3. Generates Grad-CAM heatmaps — including a deliberate test on a
     noisy-background leaf image, which is the direct evidence needed
     for Defense Question 1.
  4. Tests the OOD confidence threshold on an out-of-class image (a
     non-plant photo), which is the evidence for Defense Question 3.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, MODEL_DIR, PLOT_DIR, WORKING_DIR, OOD_CONFIDENCE_THRESHOLD
from data.dataset_loader import make_dataset


def load_model_and_classes():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "agrivision_final.keras"))
    with open(os.path.join(MODEL_DIR, "class_index.json")) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class


def evaluate_test_set(model, idx_to_class):
    test_df = pd.read_csv(os.path.join(WORKING_DIR, "test_holdout.csv"))
    test_ds = make_dataset(test_df, training=False)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    labels_sorted = sorted(idx_to_class.keys())
    names = [idx_to_class[i] for i in labels_sorted]

    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    print(report)
    with open(os.path.join(PLOT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Saved confusion matrix to {PLOT_DIR}/confusion_matrix.png")

    return cm, report


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def find_last_conv_layer(model):
    """EfficientNetB3's last conv layer is nested inside the base model
    submodule, not the outer model — this walks in to find it."""
    base = model.get_layer(index=2)  # the EfficientNetB3 functional submodule
    for layer in reversed(base.layers):
        if len(layer.output_shape) == 4:  # conv/activation layers are 4D
            return base, layer.name
    raise ValueError("No 4D conv layer found — check model architecture.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, base_model_layer_index=2):
    base = model.get_layer(index=base_model_layer_index)
    grad_model = tf.keras.models.Model(
        inputs=base.inputs,
        outputs=[base.get_layer(last_conv_layer_name).output, base.output],
    )
    # Rebuild a mini forward path through the head layers after the base model
    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(img_array)
        x = base_out
        for layer in model.layers[3:]:
            x = layer(x)
        pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def overlay_gradcam(img_path, heatmap, alpha=0.4, save_path=None):
    import cv2
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img).astype("uint8")

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (heatmap * alpha + img * (1 - alpha)).astype("uint8")
    if save_path:
        plt.imsave(save_path, overlay)
    return overlay


def test_gradcam_on_sample(model, idx_to_class, image_path, label="sample"):
    """Point this at a specific image from your test set — deliberately
    pick one with a messy/noisy field background to generate the exact
    evidence Defense Question 1 asks for. Compare: does the heatmap sit
    on the lesion, or does it light up background regions?"""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, pred_idx = make_gradcam_heatmap(img_array, model, "top_conv")
    pred_class = idx_to_class[pred_idx]

    save_path = os.path.join(PLOT_DIR, f"gradcam_{label}.png")
    overlay_gradcam(image_path, heatmap, save_path=save_path)
    print(f"[{label}] Predicted: {pred_class} | Grad-CAM saved to {save_path}")
    print("  -> Manually inspect: does the highlighted region sit on the "
          "lesion, or on background/soil? This IS your Defense Q1 evidence.")
    return pred_class, save_path


# ---------------------------------------------------------------------------
# OOD threshold test
# ---------------------------------------------------------------------------
def test_ood_rejection(model, idx_to_class, non_plant_image_path):
    """Feed in a photo of something that isn't a leaf at all (e.g. a shoe,
    a car, a random object). A well-behaved model should produce a LOW
    max-softmax confidence here. If it confidently predicts a disease
    class, OOD_CONFIDENCE_THRESHOLD in config.py needs raising, or you
    need a more deliberate OOD strategy — either way, this test result
    IS your Defense Question 3 evidence."""
    img = tf.keras.utils.load_img(non_plant_image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    max_conf = float(np.max(preds))
    pred_class = idx_to_class[int(np.argmax(preds))]

    if max_conf < OOD_CONFIDENCE_THRESHOLD:
        print(f"OOD test PASSED: max confidence {max_conf:.2f} < threshold "
              f"{OOD_CONFIDENCE_THRESHOLD} -> app would report 'uncertain'.")
    else:
        print(f"OOD test FAILED: model predicted '{pred_class}' at "
              f"{max_conf:.2f} confidence on a non-plant image. Consider "
              f"raising OOD_CONFIDENCE_THRESHOLD or adding temperature "
              f"scaling / an explicit 'not a leaf' rejection class.")
    return max_conf, pred_class


if __name__ == "__main__":
    model, class_to_idx, idx_to_class = load_model_and_classes()
    evaluate_test_set(model, idx_to_class)
    # Fill in real paths to a messy-background test image and a non-plant
    # image before running these two lines:
    # test_gradcam_on_sample(model, idx_to_class, "/path/to/messy_bg_leaf.jpg", "noisy_bg")
    # test_ood_rejection(model, idx_to_class, "/path/to/non_plant_photo.jpg")
