"""
data/dataset_loader.py — AgriVision AI

What this file does:
1. Reads the Cassava competition CSV + PlantVillage folder structure.
2. Builds ONE unified manifest (filepath, crop, class_name, label_id).
3. Splits into train / val / test (stratified by class).
4. Returns tf.data.Dataset objects with augmentation baked in for training.

Run verify_paths() FIRST in a notebook cell to confirm every path resolves
before building the manifest — Kaggle dataset mirrors vary in internal
folder naming and this will save you a confusing debugging session.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CASSAVA_TRAIN_CSV, CASSAVA_IMAGE_DIR, CASSAVA_LABEL_MAP,
    PLANTVILLAGE_IMAGE_ROOTS, PLANTVILLAGE_FOLDER_MAP,
    IMG_SIZE, BATCH_SIZE, SEED, VAL_SPLIT, TEST_SPLIT, MANIFEST_PATH,
)


def verify_paths():
    """Run this in a notebook cell before anything else. Prints whether
    each expected path exists, so you catch mismatched folder names early."""
    checks = {
        "CASSAVA_TRAIN_CSV": CASSAVA_TRAIN_CSV,
        "CASSAVA_IMAGE_DIR": CASSAVA_IMAGE_DIR,
    }
    for name, path in checks.items():
        print(f"[{'OK' if os.path.exists(path) else 'MISSING'}] {name}: {path}")

    for root in PLANTVILLAGE_IMAGE_ROOTS:
        print(f"[{'OK' if os.path.exists(root) else 'MISSING'}] PLANTVILLAGE root: {root}")
        if os.path.exists(root):
            found = set(os.listdir(root))
            expected = set(PLANTVILLAGE_FOLDER_MAP.keys())
            missing = expected - found
            if missing:
                print("  Folders expected but NOT found (check exact naming):")
                for m in missing:
                    print("    -", m)
            else:
                print("  All expected class folders found.")


def _build_cassava_manifest():
    df = pd.read_csv(CASSAVA_TRAIN_CSV)
    df["filepath"] = df["image_id"].apply(lambda x: os.path.join(CASSAVA_IMAGE_DIR, x))
    df["class_name"] = df["label"].map(CASSAVA_LABEL_MAP)
    df["crop"] = "cassava"
    df = df[["filepath", "crop", "class_name"]]
    df = df[df["filepath"].apply(os.path.exists)]
    return df


def _build_plantvillage_manifest():
    """Pools images from BOTH the dataset's train/ and valid/ folders —
    we re-split everything ourselves in split_manifest() rather than
    relying on the dataset's own pre-made split."""
    rows = []
    for root in PLANTVILLAGE_IMAGE_ROOTS:
        for folder_name, class_name in PLANTVILLAGE_FOLDER_MAP.items():
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            crop = class_name.split("_")[0]
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({
                        "filepath": os.path.join(folder_path, fname),
                        "crop": crop,
                        "class_name": class_name,
                    })
    return pd.DataFrame(rows)


def build_manifest(save=True):
    """Combine both sources into one manifest with integer label_id column.
    Saves to MANIFEST_PATH so evaluate.py and the app can reuse the same
    class-index mapping without re-scanning folders."""
    cassava_df = _build_cassava_manifest()
    pv_df = _build_plantvillage_manifest()
    df = pd.concat([cassava_df, pv_df], ignore_index=True)

    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["class_name"].map(class_to_idx)

    print(f"Total images: {len(df)}")
    print(f"Total classes: {len(classes)}")
    print(df["class_name"].value_counts())

    if save:
        df.to_csv(MANIFEST_PATH, index=False)
        # also save the class mapping — evaluate.py and app.py both need this
        pd.Series(class_to_idx).to_json(MANIFEST_PATH.replace(".csv", "_classes.json"))

    return df, class_to_idx


def split_manifest(df):
    """Stratified train/val/test split. Test set is held out and only
    touched once, at the very end, in evaluate.py."""
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=df["label_id"], random_state=SEED
    )
    relative_test = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label_id"], random_state=SEED
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(train_df, num_classes):
    """PlantVillage's tomato subset alone has a >14:1 imbalance ratio between
    its largest and smallest class. Without this, the model will just learn
    to predict the majority classes and still look 'accurate' on paper."""
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_df["label_id"].values,
    )
    return dict(enumerate(weights))


# ---------------------------------------------------------------------------
# Augmentation — this is the block most relevant to Defense Question 1
# (background shortcut learning). PlantVillage images have plain/lab
# backgrounds; Cassava images have messy real field backgrounds. Without
# strong augmentation the model can learn "plain background -> PlantVillage
# classes" as a shortcut instead of actually reading the lesion.
# ---------------------------------------------------------------------------
def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        # RandomTranslation forces the model to see the leaf/lesion at
        # different frame positions rather than always centered — this
        # specifically discourages background-position shortcuts.
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # Coarse dropout-style occlusion (manual "cutout"): randomly zeroes
        # a patch of the image so the model can't rely on any single fixed
        # region (including background corners) always being informative.
    ], name="augmentation")


def _cutout(image, patch_frac=0.15):
    h, w = IMG_SIZE
    ph, pw = int(h * patch_frac), int(w * patch_frac)
    top = tf.random.uniform([], 0, h - ph, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - pw, dtype=tf.int32)
    mask = tf.pad(
        tf.zeros((ph, pw, 3)),
        [[top, h - top - ph], [left, w - left - pw], [0, 0]],
        constant_values=1,
    )
    return image * mask


def _load_image(filepath, label, training):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    if training:
        img = _cutout(img)
    return img, label


def make_dataset(df, training=True, augment_layer=None):
    filepaths = df["filepath"].values
    labels = df["label_id"].values
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.map(lambda f, l: _load_image(f, l, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if training and augment_layer is not None:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    verify_paths()
