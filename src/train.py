"""
Train the attention-based captioning model.

Improvements over the original project's training loop:
- Proper masked loss (padding tokens don't contribute to the loss).
- tf.function-compiled train_step for speed.
- Checkpoint manager that keeps the last N checkpoints and can resume.
- Saves a loss-curve plot instead of just printing numbers.
"""

import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

import config
import data_prep
import dataset as ds_module
from caption_model import CNN_Encoder, RNN_Decoder

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


@tf.function
def train_step(img_tensor, target, encoder, decoder, optimizer, tokenizer):
    loss = 0
    batch_size = target.shape[0]
    hidden = decoder.reset_state(batch_size=batch_size)

    start_token_id = tokenizer.word_index[config.START_TOKEN]
    dec_input = tf.expand_dims([start_token_id] * batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += masked_loss(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)  # teacher forcing

    total_loss = loss / int(target.shape[1])
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss, total_loss


def main():
    train_descriptions, _, tokenizer, max_len = data_prep.prepare_all()

    vocab_size = min(config.TOP_K_VOCAB, len(tokenizer.word_index)) + 1

    train_ds = ds_module.build_dataset(train_descriptions, tokenizer, max_len)
    num_steps = sum(len(c) for c in train_descriptions.values()) // config.BATCH_SIZE

    encoder = CNN_Encoder(config.EMBEDDING_DIM)
    decoder = RNN_Decoder(config.EMBEDDING_DIM, config.UNITS, vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, config.CHECKPOINT_DIR, max_to_keep=3)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Resumed from checkpoint, starting at epoch {start_epoch}")

    loss_plot = []

    for epoch in range(start_epoch, config.EPOCHS):
        start = time.time()
        total_loss = 0

        for batch, (img_tensor, target) in enumerate(train_ds):
            batch_loss, t_loss = train_step(
                img_tensor, target, encoder, decoder, optimizer, tokenizer
            )
            total_loss += t_loss

            if batch % 50 == 0:
                avg = batch_loss.numpy() / int(target.shape[1])
                print(f"Epoch {epoch+1} Batch {batch} Loss {avg:.4f}")

        loss_plot.append(total_loss / max(num_steps, 1))
        ckpt_manager.save()
        print(f"Epoch {epoch+1} Loss {loss_plot[-1]:.6f} Time {time.time()-start:.1f}s")

    plt.figure()
    plt.plot(loss_plot)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(config.LOSS_PLOT_PATH)
    print(f"Saved loss curve to {config.LOSS_PLOT_PATH}")

    with open(config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)


if __name__ == "__main__":
    main()
