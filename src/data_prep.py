"""
Load, clean, and tokenize the Flickr8k captions.

Improvements over the original script this project started from:
- No hardcoded personal paths (everything comes from config.py).
- Only keeps the TOP_K_VOCAB most frequent words instead of the full
  (noisy, typo-ridden) vocabulary -> smaller, better-trained embeddings.
- Adds explicit <start>/<end> tokens and an <unk> fallback so inference
  on real, messy input doesn't crash on unseen words.
"""

import os
import string
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

import config


def load_doc(filename):
    with open(filename, "r") as f:
        return f.read()


def all_img_captions(token_filename):
    """Parse Flickr8k.token.txt into {image_filename: [captions]}."""
    text = load_doc(token_filename)
    descriptions = {}
    for line in text.strip().split("\n"):
        if "\t" not in line:
            continue
        img_id, caption = line.split("\t")
        img_file = img_id.split("#")[0]
        descriptions.setdefault(img_file, []).append(caption)
    return descriptions


def clean_captions(descriptions):
    """Lowercase, strip punctuation, drop numeric/single-char tokens."""
    table = str.maketrans("", "", string.punctuation)
    cleaned = {}
    for img, caps in descriptions.items():
        cleaned[img] = []
        for cap in caps:
            words = cap.replace("-", " ").split()
            words = [w.lower() for w in words]
            words = [w.translate(table) for w in words]
            words = [w for w in words if len(w) > 1 and w.isalpha()]
            cleaned[img].append(" ".join(words))
    return cleaned


def add_start_end_tokens(descriptions):
    out = {}
    for img, caps in descriptions.items():
        out[img] = [f"{config.START_TOKEN} {c} {config.END_TOKEN}" for c in caps]
    return out


def save_descriptions(descriptions, filename):
    lines = [f"{img}\t{cap}" for img, caps in descriptions.items() for cap in caps]
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def load_clean_descriptions(filename, photo_ids):
    text = load_doc(filename)
    descriptions = {}
    for line in text.split("\n"):
        if "\t" not in line:
            continue
        img, cap = line.split("\t", 1)
        if img in photo_ids:
            descriptions.setdefault(img, []).append(cap)
    return descriptions


def load_photo_ids(split_filename, images_dir):
    text = load_doc(split_filename)
    ids = [line.strip() for line in text.split("\n") if line.strip()]
    # only keep ones that actually exist on disk
    return [i for i in ids if os.path.exists(os.path.join(images_dir, i))]


def dict_to_list(descriptions):
    return [c for caps in descriptions.values() for c in caps]


def build_tokenizer(descriptions):
    """Fit a Keras Tokenizer restricted to the TOP_K_VOCAB most frequent words."""
    all_caps = dict_to_list(descriptions)
    tokenizer = Tokenizer(
        num_words=config.TOP_K_VOCAB,
        oov_token=config.UNK_TOKEN,
        filters="",  # captions are already cleaned; don't strip <start>/<end>
    )
    tokenizer.fit_on_texts(all_caps)
    # Keras reserves index 0 for padding; make sure that's respected.
    tokenizer.word_index[tokenizer.oov_token] = tokenizer.word_index.get(
        tokenizer.oov_token, config.TOP_K_VOCAB + 1
    )
    return tokenizer


def max_caption_length(descriptions):
    return max(len(c.split()) for c in dict_to_list(descriptions))


def prepare_all():
    """
    Full data prep pipeline. Returns (train_descriptions, test_descriptions,
    tokenizer, max_len).
    """
    print("Parsing raw captions...")
    descriptions = all_img_captions(config.TOKEN_FILE)
    print(f"  {len(descriptions)} images with captions found")

    print("Cleaning captions...")
    descriptions = clean_captions(descriptions)
    descriptions = add_start_end_tokens(descriptions)
    save_descriptions(descriptions, config.CLEAN_CAPTIONS_PATH)

    train_ids = load_photo_ids(config.TRAIN_SPLIT_FILE, config.DATASET_IMAGES_DIR)
    test_ids = load_photo_ids(config.TEST_SPLIT_FILE, config.DATASET_IMAGES_DIR)
    print(f"  train images: {len(train_ids)} | test images: {len(test_ids)}")

    train_descriptions = load_clean_descriptions(config.CLEAN_CAPTIONS_PATH, set(train_ids))
    test_descriptions = load_clean_descriptions(config.CLEAN_CAPTIONS_PATH, set(test_ids))

    tokenizer = build_tokenizer(train_descriptions)
    with open(config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    max_len = max_caption_length(train_descriptions)
    vocab_size = min(config.TOP_K_VOCAB, len(tokenizer.word_index)) + 1
    print(f"  vocab size: {vocab_size} | max caption length: {max_len}")

    return train_descriptions, test_descriptions, tokenizer, max_len


if __name__ == "__main__":
    prepare_all()
