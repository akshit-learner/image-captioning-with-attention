"""
Inference on new, unseen images -- the piece that was completely missing
from the original project. Supports:
  - greedy decoding
  - beam search decoding
  - attention-map visualization (what the model "looked at" per word)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config
from caption_model import CNN_Encoder, RNN_Decoder
from features import build_encoder_cnn, load_and_preprocess_image


class CaptionPredictor:
    def __init__(self, checkpoint_dir=None, tokenizer_path=None):
        checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        tokenizer_path = tokenizer_path or config.TOKENIZER_PATH

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        vocab_size = min(config.TOP_K_VOCAB, len(self.tokenizer.word_index)) + 1

        self.cnn = build_encoder_cnn()
        self.encoder = CNN_Encoder(config.EMBEDDING_DIM)
        self.decoder = RNN_Decoder(config.EMBEDDING_DIM, config.UNITS, vocab_size)

        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. Train the model first."
            )
        ckpt.restore(latest).expect_partial()

        self.start_id = self.tokenizer.word_index[config.START_TOKEN]
        self.end_id = self.tokenizer.word_index[config.END_TOKEN]
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}

    def _extract_features(self, image_path):
        img, _ = load_and_preprocess_image(image_path)
        img = tf.expand_dims(img, 0)
        feat = self.cnn(img)  # (1, 8, 8, 2048)
        feat = tf.reshape(feat, (1, -1, feat.shape[-1]))  # (1, 64, 2048)
        return self.encoder(feat)  # (1, 64, embedding_dim)

    def generate_greedy(self, image_path, max_len=None):
        max_len = max_len or config.MAX_CAPTION_LEN
        features = self._extract_features(image_path)
        hidden = self.decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([self.start_id], 0)

        result, attention_plots = [], []
        for _ in range(max_len):
            preds, hidden, attn = self.decoder(dec_input, features, hidden)
            attention_plots.append(tf.reshape(attn, (-1,)).numpy())
            predicted_id = tf.argmax(preds[0]).numpy()

            if predicted_id == self.end_id:
                break
            word = self.index_word.get(predicted_id, config.UNK_TOKEN)
            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)

        return " ".join(result), attention_plots

    def generate_beam_search(self, image_path, beam_width=None, max_len=None):
        """Standard beam search: keep top-k partial sequences by
        cumulative log-probability at every step."""
        beam_width = beam_width or config.BEAM_WIDTH
        max_len = max_len or config.MAX_CAPTION_LEN
        features = self._extract_features(image_path)

        # each beam entry: (token_ids list, log_prob, hidden_state)
        init_hidden = self.decoder.reset_state(batch_size=1)
        beams = [([self.start_id], 0.0, init_hidden)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, log_prob, hidden in beams:
                if seq[-1] == self.end_id:
                    completed.append((seq, log_prob))
                    continue
                dec_input = tf.expand_dims([seq[-1]], 0)
                preds, new_hidden, _ = self.decoder(dec_input, features, hidden)
                log_probs = tf.nn.log_softmax(preds[0]).numpy()
                top_ids = np.argsort(log_probs)[-beam_width:]

                for token_id in top_ids:
                    candidates.append(
                        (seq + [int(token_id)], log_prob + log_probs[token_id], new_hidden)
                    )

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        completed.extend([(s, lp) for s, lp, _ in beams])
        # normalize by length to avoid favoring short captions
        completed.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
        best_seq = completed[0][0]

        words = [
            self.index_word.get(t, config.UNK_TOKEN)
            for t in best_seq
            if t not in (self.start_id, self.end_id)
        ]
        return " ".join(words)

    def caption(self, image_path, method="beam"):
        if method == "greedy":
            text, _ = self.generate_greedy(image_path)
            return text
        return self.generate_beam_search(image_path)


def plot_attention(image_path, result_words, attention_plots, save_path=None):
    """Overlay each timestep's attention map (8x8, upsampled) on the image."""
    img = plt.imread(image_path)
    n = len(result_words)
    cols = min(5, max(1, n))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i, word in enumerate(result_words):
        attn = attention_plots[i].reshape(8, 8)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(word)
        ax.imshow(img)
        ax.imshow(attn, cmap="jet", alpha=0.5, extent=(0, img.shape[1], img.shape[0], 0))
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [greedy|beam]")
        sys.exit(1)

    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "beam"

    predictor = CaptionPredictor()
    caption = predictor.caption(image_path, method=method)
    print(f"Caption ({method}): {caption}")
