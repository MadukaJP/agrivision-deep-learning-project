"""
config.py — Central configuration for AgriVision AI

Run the folder-listing snippet from Step 3 first (os.walk on /kaggle/input)
and adjust CASSAVA_DIR / PLANTVILLAGE_DIR below to match what you actually
see. Kaggle dataset mount paths are stable but PlantVillage mirrors differ
in their internal folder naming, so DO NOT assume these paths are correct
until you've confirmed them against your own notebook's output.
"""

import os

# ---------------------------------------------------------------------------
# PATHS — adjust these after running the os.walk verification step
# ---------------------------------------------------------------------------
KAGGLE_INPUT = "/kaggle/input"

# Cassava Leaf Disease Classification competition dataset
# Note: Kaggle nests competition datasets under /kaggle/input/competitions/
CASSAVA_DIR = os.path.join(KAGGLE_INPUT, "competitions", "cassava-leaf-disease-classification")
CASSAVA_TRAIN_CSV = os.path.join(CASSAVA_DIR, "train.csv")
CASSAVA_IMAGE_DIR = os.path.join(CASSAVA_DIR, "train_images")
CASSAVA_LABEL_MAP = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mottle",
    3: "cassava_mosaic_disease",
    4: "cassava_healthy",
}

# "New Plant Diseases Dataset" (vipoooool) — covers maize, tomato, pepper.
# Nested under /kaggle/input/datasets/vipoooool/... on Kaggle, with the
# folder name doubled (this is normal for Kaggle-mirrored datasets).
# Ships pre-split into train/ and valid/ folders, each with the same class
# subfolders — we pool BOTH together and let our own split_manifest() do
# a fresh stratified split across everything (cassava + this).
PLANTVILLAGE_DIR = os.path.join(
    KAGGLE_INPUT, "datasets", "vipoooool", "new-plant-diseases-dataset",
    "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)",
)
PLANTVILLAGE_IMAGE_ROOTS = [
    os.path.join(PLANTVILLAGE_DIR, "train"),
    os.path.join(PLANTVILLAGE_DIR, "valid"),
]

# Which PlantVillage folder-name prefixes map to which crop, and how we
# rename each raw folder name into our unified class_name scheme.
PLANTVILLAGE_FOLDER_MAP = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "maize_northern_leaf_blight",
    "Corn_(maize)___healthy": "maize_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___healthy": "tomato_healthy",
    "Pepper,_bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper,_bell___healthy": "pepper_healthy",
}

# ---------------------------------------------------------------------------
# OUTPUT PATHS — everything written here survives as a Kaggle "Output" once
# you commit/save the notebook. Download it before the session ends.
# ---------------------------------------------------------------------------
WORKING_DIR = "/kaggle/working"
MODEL_DIR = os.path.join(WORKING_DIR, "models", "saved_models")
LOG_DIR = os.path.join(WORKING_DIR, "models", "logs")
PLOT_DIR = os.path.join(WORKING_DIR, "models", "plots")
MANIFEST_PATH = os.path.join(WORKING_DIR, "dataset_manifest.csv")

for d in [MODEL_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# MODEL / TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
IMG_SIZE = (300, 300)          # EfficientNetB3 native input size
BATCH_SIZE = 16                # keep modest — B3 at 300x300 is memory-heavy on Kaggle GPUs
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10              # held out from val, never touched until final evaluate.py run

PHASE1_EPOCHS = 12
PHASE1_LR = 1e-3

PHASE2_EPOCHS = 20
PHASE2_LR = 1e-5
PHASE2_UNFREEZE_LAYERS = 40     # unfreeze top N layers of the backbone

# Out-of-distribution rejection: below this softmax confidence, the app
# reports "uncertain / not a recognized crop-disease" instead of forcing
# a class. This is the OOD mechanism for Defense Question 3 (threshold-based,
# not a separate background class — see DEFENSE_ANSWERS notes).
OOD_CONFIDENCE_THRESHOLD = 0.60
