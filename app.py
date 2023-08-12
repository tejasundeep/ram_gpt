import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Read and preprocess the text dataset
with open('dataset.txt', 'r', encoding='utf-8') as file:
    raw_text = [text.lower().strip() for text in file.read().splitlines()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_text)
vocab_size = len(tokenizer.word_index) + 1

text_sequences = tokenizer.texts_to_sequences(raw_text)
max_seq_len = 100
padded_sequences = pad_sequences(text_sequences, maxlen=max_seq_len, padding='post', truncating='post')

# Positional encoding function
def create_positional_encoding(position, model_dim):
    angle_rads = position / np.power(10000, 2 * (np.arange(model_dim) // 2) / model_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

# Multi-head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):
    # The `__init__` function initializes the parameters and layers for a multi-head attention mechanism.
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj_dim = embed_dim // self.num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    # The function splits the input tensor into multiple heads by reshaping and transposing it.
    def split_into_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    # The function performs self-attention on the input tensor and returns the combined attention output.
    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        query, key, value = self.query_dense(inputs), self.key_dense(inputs), self.value_dense(inputs)
        query, key, value = self.split_into_heads(query, batch_size), self.split_into_heads(key, batch_size), self.split_into_heads(value, batch_size)

        scaled_attention = tf.matmul(query, key, transpose_b=True)
        if mask is not None:
            scaled_attention += (mask * -1e9)

        scaled_attention /= tf.math.sqrt(tf.cast(self.proj_dim, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(attention_output)

# Self-attention layer with multi-head attention
class SelfAttention(tf.keras.layers.Layer):
    # The function initializes a self-attention layer with a specified embedding dimension and number of attention heads.
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.output_transform = tf.keras.layers.Dense(embed_dim)

    # The function performs multi-head attention on the input and returns the transformed output.
    def call(self, inputs, mask=None):
        attention_output = self.multi_head_attention(inputs, mask=mask)
        return self.output_transform(attention_output)

# Transformer layer
class TransformerLayer(tf.keras.layers.Layer):
    # The above function defines a Transformer layer in a neural network model for natural language processing tasks.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = tf.convert_to_tensor([create_positional_encoding(pos, embed_dim) for pos in range(max_seq_len)], dtype=tf.float32)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    # The function takes inputs, applies self-attention and feed-forward neural network layers, and returns the output.
    def call(self, inputs, mask=None):
        word_emb = self.embedding(inputs) * tf.math.sqrt(tf.cast(self.embedding.units, tf.float32))
        pos_emb = word_emb + self.pos_encoding
        pos_emb = self.dropout(pos_emb)

        attention_output = self.self_attention(pos_emb, mask=mask)
        attention_output = self.layer_norm1(attention_output + pos_emb)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.layer_norm2(ffn_output + attention_output)

        return self.output_layer(ffn_output)

# Transformer model
class CustomTransformer(tf.keras.Model):
    # The `__init__` function initializes a `CustomTransformer` object with an encoder and decoder, both of which are instances of the `TransformerLayer` class.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(CustomTransformer, self).__init__()
        self.encoder = TransformerLayer(embed_dim, num_heads, ff_dim, rate)
        self.decoder = TransformerLayer(embed_dim, num_heads, ff_dim, rate)

    # The function takes inputs and an optional mask, passes the inputs through an encoder and a decoder, and returns the output of the decoder.
    def call(self, inputs, mask=None):
        enc_output = self.encoder(inputs, mask=mask)
        dec_output = self.decoder(inputs, mask=mask)
        return dec_output

# Model setup and training
embedding_dim = 256
num_attention_heads = 4
feed_forward_dim = 512

transformer_model = CustomTransformer(embedding_dim, num_attention_heads, feed_forward_dim)
optimizer = tf.keras.optimizers.Adam()

target_data = padded_sequences[:, 1:]
mask = 1 - tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0)
attention_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
attention_mask = tf.tile(attention_mask[tf.newaxis, tf.newaxis, :, :], [target_data.shape[0], 1, 1, 1])

batch_size = 64
accumulation_steps = 4

# The code snippet is training the Transformer model using mini-batch gradient descent. It iterates
# over the dataset for a specified number of epochs and for each epoch, it divides the dataset into
# mini-batches of size `batch_size * accumulation_steps`.
for epoch in range(10):
    for i in range(0, len(padded_sequences), batch_size * accumulation_steps):
        batch_inputs = padded_sequences[i:i + batch_size * accumulation_steps]
        batch_targets = target_data[i:i + batch_size * accumulation_steps]
        batch_mask = attention_mask[i:i + batch_size * accumulation_steps]

        with tf.GradientTape() as tape:
            preds = transformer_model(batch_inputs, mask=batch_mask, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch_targets, preds, from_logits=True))

        gradients = tape.gradient(loss, transformer_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))

        if (i // batch_size + 1) % accumulation_steps == 0:
            print(f"Epoch {epoch+1}, Step {i//batch_size + 1}, Loss: {loss.numpy()}")

# Text generation function
def generate_text(initial_text, max_length):
    """
    The function `generate_text` takes an initial text and a maximum length as input, and generates text
    by predicting the next word using a transformer model until either the '<end>' token is encountered
    or the maximum length is reached.
    """
    while '<end>' not in initial_text and len(initial_text.split()) < max_length:
        initial_seq = tokenizer.texts_to_sequences([initial_text])[0]
        padded_initial = pad_sequences([initial_seq], maxlen=max_seq_len, padding='post', truncating='post')
        pred_id = np.argmax(transformer_model.predict(padded_initial), axis=-1)[:, -1]
        pred_word = tokenizer.index_word.get(pred_id[0][-1], '<unknown>')
        initial_text += " " + pred_word
    return initial_text

# Generate and print text
seed_text = "once upon a time"
generated_text = generate_text(seed_text, max_length=50)
print("Generated Text:", generated_text)
