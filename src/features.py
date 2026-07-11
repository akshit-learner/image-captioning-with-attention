"""
Extract spatial CNN feature maps for every image, used later by the
attention decoder.

Key difference from the original project: instead of pooling each image
down to a single 2048-d vector (which throws away *where* things are in
the image), we keep the last convolutional feature map (8x8x2048 for
InceptionV3) and flatten it to a (64, 2048) sequence of "visual words".
The attention layer then learns which of those 64 spatial locations to
look at for each output word. This is what makes attention maps /
heatmaps possible in predict.py.
"""

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

import config


def build_encoder_cnn():
    """InceptionV3 with the classification head removed, stopping at the
    last conv block so we keep spatial structure: output shape (8, 8, 2048)."""
    base = InceptionV3(include_top=False, weights="imagenet")
    return tf.keras.Model(inputs=base.input, outputs=base.output)


def load_and_preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = preprocess_input(img)
    return img, img_path


def extract_and_cache_features(image_filenames, images_dir, features_dir, batch_size=16):
    """
    Runs InceptionV3 over every image and saves each (64, 2048) feature
    map to disk as {features_dir}/{image_filename}.npy so training can
    just memory-map them instead of re-running the CNN every epoch.
    """
    os.makedirs(features_dir, exist_ok=True)
    already_done = set(os.listdir(features_dir))
    todo = [
        f for f in image_filenames
        if f"{f}.npy" not in already_done
    ]
    if not todo:
        print("All features already cached, skipping extraction.")
        return

    print(f"Extracting features for {len(todo)} images...")
    encoder = build_encoder_cnn()

    full_paths = [os.path.join(images_dir, f) for f in todo]
    ds = tf.data.Dataset.from_tensor_slices(full_paths)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    for img_batch, path_batch in tqdm(ds, total=len(todo) // batch_size + 1):
        feats = encoder(img_batch)  # (batch, 8, 8, 2048)
        feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[-1]))  # (batch, 64, 2048)
        for feat, path in zip(feats.numpy(), path_batch.numpy()):
            img_filename = os.path.basename(path.decode("utf-8"))
            out_path = os.path.join(features_dir, f"{img_filename}.npy")
            np.save(out_path, feat)


if __name__ == "__main__":
    import data_prep

    descriptions = data_prep.all_img_captions(config.TOKEN_FILE)
    all_images = list(descriptions.keys())
    extract_and_cache_features(all_images, config.DATASET_IMAGES_DIR, config.FEATURES_DIR)
