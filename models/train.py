"""
models/train.py — AgriVision AI

Two-phase transfer learning:
  Phase 1: frozen EfficientNetB3 backbone, train only the classifier head.
  Phase 2: unfreeze top N backbone layers, fine-tune at a much lower LR.

Run this AFTER dataset_loader.build_manifest() + split_manifest() have
produced train_df / val_df. Each phase saves its own history so you have
real numbers for Defense Question 2 (exact epoch overfitting began, and
what LR resolved it) — don't skip saving these, you'll need the plot.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMG_SIZE, MODEL_DIR, LOG_DIR, PLOT_DIR,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR, PHASE2_UNFREEZE_LAYERS,
)
from data.dataset_loader import (
    build_manifest, split_manifest, make_dataset,
    compute_class_weights, build_augmentation,
)


def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(*IMG_SIZE, 3), pooling="avg",
    )
    base.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def get_callbacks(phase_name):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, f"agrivision_{phase_name}_best.keras"),
            monitor="val_loss", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"{phase_name}_history.csv")
        ),
    ]


def plot_history(history, phase_name):
    """Saves the exact plot you'll show for Defense Question 2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title(f"{phase_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train_acc")
    axes[1].plot(history.history["val_accuracy"], label="val_acc")
    axes[1].set_title(f"{phase_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    path = os.path.join(PLOT_DIR, f"{phase_name}_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Flag the exact epoch val_loss stopped improving — this IS your
    # Defense Question 2 answer, read directly off real numbers.
    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1
    print(f"[{phase_name}] Best val_loss at epoch {best_epoch} "
          f"(val_loss={min(val_losses):.4f}). Epochs after this are overfitting.")


def main():
    df, class_to_idx = build_manifest()
    num_classes = len(class_to_idx)
    train_df, val_df, test_df = split_manifest(df)
    test_df.to_csv(os.path.join(os.path.dirname(MODEL_DIR), "test_holdout.csv"), index=False)

    class_weights = compute_class_weights(train_df, num_classes)
    augment = build_augmentation()

    train_ds = make_dataset(train_df, training=True, augment_layer=augment)
    val_ds = make_dataset(val_df, training=False)

    model, base = build_model(num_classes)

    # ---------------- PHASE 1: frozen backbone ----------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== PHASE 1: training classifier head (backbone frozen) ===")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS,
        class_weight=class_weights, callbacks=get_callbacks("phase1"),
    )
    plot_history(history1, "phase1")

    # ---------------- PHASE 2: fine-tune top layers ----------------
    base.trainable = True
    for layer in base.layers[:-PHASE2_UNFREEZE_LAYERS]:
        layer.trainable = False

    # Cosine annealing LR schedule for Phase 2, as specified in the brief
    steps_per_epoch = len(train_df) // 16
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=PHASE2_LR,
        decay_steps=steps_per_epoch * PHASE2_EPOCHS,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== PHASE 2: fine-tuning top layers ===")
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS,
        class_weight=class_weights, callbacks=get_callbacks("phase2"),
    )
    plot_history(history2, "phase2")

    final_path = os.path.join(MODEL_DIR, "agrivision_final.keras")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    with open(os.path.join(MODEL_DIR, "class_index.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)


if __name__ == "__main__":
    main()
