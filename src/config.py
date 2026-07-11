"""
Central configuration for the image captioning project.
Edit the paths below to match where you've placed the Flickr8k dataset
(e.g. in Google Drive if running on Colab).
"""

import os

# ---------------------------------------------------------------------------
# Paths (override these, or set as environment variables before running)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("CAPTION_PROJECT_ROOT", os.getcwd())

# Folder containing Flickr8k_text/ (captions) and Flicker8k_Dataset/ (images)
DATASET_TEXT_DIR = os.environ.get(
    "CAPTION_TEXT_DIR", os.path.join(PROJECT_ROOT, "data", "Flickr8k_text")
)
DATASET_IMAGES_DIR = os.environ.get(
    "CAPTION_IMAGES_DIR", os.path.join(PROJECT_ROOT, "data", "Flicker8k_Dataset")
)

# Where processed artifacts (cached features, tokenizer, checkpoints) live
ARTIFACTS_DIR = os.environ.get(
    "CAPTION_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "artifacts")
)
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.p")
CLEAN_CAPTIONS_PATH = os.path.join(ARTIFACTS_DIR, "captions_clean.txt")
LOSS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "loss_plot.png")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "eval_results.json")

for _dir in [ARTIFACTS_DIR, FEATURES_DIR, CHECKPOINT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Raw Flickr8k split files
TRAIN_SPLIT_FILE = os.path.join(DATASET_TEXT_DIR, "Flickr_8k.trainImages.txt")
DEV_SPLIT_FILE = os.path.join(DATASET_TEXT_DIR, "Flickr_8k.devImages.txt")
TEST_SPLIT_FILE = os.path.join(DATASET_TEXT_DIR, "Flickr_8k.testImages.txt")
TOKEN_FILE = os.path.join(DATASET_TEXT_DIR, "Flickr8k.token.txt")

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE = 299                 # InceptionV3 input size
FEATURE_SHAPE = (64, 2048)     # 8x8 spatial grid flattened, 2048 channels

TOP_K_VOCAB = 5000             # keep the most frequent 5000 words
EMBEDDING_DIM = 256
UNITS = 512                    # GRU hidden size / attention units

BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 20

BEAM_WIDTH = 3
MAX_CAPTION_LEN = 40           # hard safety cap for generation loop

START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
