"""
Builds the tf.data.Dataset used for training: pairs of
(cached image feature map, tokenized/padded caption).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config


def descriptions_to_pairs(descriptions):
    """Flatten {image: [captions]} into parallel (image, caption) lists."""
    img_names, caps = [], []
    for img, cap_list in descriptions.items():
        for cap in cap_list:
            img_names.append(img)
            caps.append(cap)
    return img_names, caps


def tokenize_captions(captions, tokenizer, max_len):
    seqs = tokenizer.texts_to_sequences(captions)
    return pad_sequences(seqs, maxlen=max_len, padding="post")


def _load_feature(img_name):
    path = config.FEATURES_DIR + "/" + img_name.numpy().decode("utf-8") + ".npy"
    return np.load(path).astype(np.float32)


def _tf_load_feature(img_name, cap):
    feature = tf.py_function(_load_feature, [img_name], tf.float32)
    feature.set_shape(config.FEATURE_SHAPE)
    return feature, cap


def build_dataset(descriptions, tokenizer, max_len, batch_size=None, shuffle=True):
    img_names, caps = descriptions_to_pairs(descriptions)
    cap_seqs = tokenize_captions(caps, tokenizer, max_len)

    ds = tf.data.Dataset.from_tensor_slices((img_names, cap_seqs))
    ds = ds.map(_tf_load_feature, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(config.BUFFER_SIZE)

    ds = ds.batch(batch_size or config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
