# ======================================
# TRAINING CONFIG
# ======================================

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

EPOCHS = 10

IMAGE_SIZE = 224

SEED = 42

# ======================================
# PATHS
# ======================================

TRAIN_DIR = "data/chest_xray/train"

VAL_DIR = "data/chest_xray/val"

TEST_DIR = "data/chest_xray/test"

MODEL_PATH = "models/best_model.pth"

OUTPUT_DIR = "outputs"

# ======================================
# CLASS NAMES
# ======================================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]