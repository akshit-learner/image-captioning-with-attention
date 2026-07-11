"""
Attention-based image captioning model (Bahdanau/"Show, Attend and Tell"
style), replacing the original project's single-vector CNN-feature +
plain-LSTM merge architecture.

Why this is better than the merge-model:
- The decoder can look at *different regions* of the image for each word
  it generates ("dog" -> attends to the dog region, "grass" -> attends to
  the ground), instead of squashing the whole image into one static
  256-d vector reused for every timestep.
- Produces interpretable attention maps (see predict.py) which is both
  a debugging tool and a nice demo/README visual.
"""

import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    """Computes a context vector as a weighted sum over encoder features,
    where weights depend on the decoder's current hidden state."""

    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, 64, embedding_dim)
        # hidden:   (batch, units)  -- decoder's previous hidden state
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch, 1, units)

        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch, 64, 1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, embedding_dim)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    """Projects the (64, 2048) CNN feature map down to (64, embedding_dim)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    """GRU decoder with attention. Generates one word at a time,
    conditioned on the attended image context + previous word."""

    def __init__(self, embedding_dim, units, vocab_size):
        super().__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden):
        # x: (batch, 1) -- the previous word's token id
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)  # (batch, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)  # (batch, vocab_size)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
