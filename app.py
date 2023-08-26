import tensorflow as tf
import numpy as np

# Scaled dot-product attention
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * tf.constant(-np.inf, dtype=tf.float32))

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

# Custom Loss Function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# Create Padding Mask
def create_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

# Text Processor
class TextProcessor:
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        
    def fit_tokenizer(self, dummy_sentences):
        self.tokenizer.fit_on_texts(dummy_sentences)
    
    def preprocess_text(self, text):
        return tf.strings.lower(text)
    
    def tokenize_map_fn(self, text):
        text = self.tokenizer.texts_to_sequences([text.numpy().decode('utf-8')])[0]
        return text[:-1], text[1:]
    
    def prepare_dataset(self, text_data):
        text_data = text_data.map(
            lambda text: tf.py_function(func=self.tokenize_map_fn, inp=[text], Tout=(tf.int64, tf.int64))
        )
        text_data = text_data.map(
            lambda x, y: (
                tf.pad(x, paddings=[[0, self.max_seq_len - tf.shape(x)[0]]], constant_values=0),
                tf.pad(y, paddings=[[0, self.max_seq_len - tf.shape(y)[0]]], constant_values=0)
            )
        )
        return text_data

# Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)
        output = scaled_dot_product_attention(query, key, value, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output

# Positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_embeddings = self.add_weight(
            shape=(position, d_model),
            initializer="uniform",
            trainable=True
        )
        
    def call(self, inputs):
        return inputs + self.positional_embeddings

# Decoder Layer with Dropout
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Model Building
def build_model(vocab_size, d_model, num_heads, dff, num_layers, dropout_rate):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int64)
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    x = PositionalEncoding(position=max_seq_len, d_model=d_model)(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    
    mask = create_padding_mask(inputs)
    
    for _ in range(num_layers):
        x = DecoderLayer(d_model, num_heads, dff, dropout_rate)(x, training=True, mask=mask)
    
    outputs = tf.keras.layers.Dense(vocab_size, use_bias=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Training
def train_model(model, train_dataset, epochs):
    optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    model.fit(train_dataset, epochs=epochs)

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 50057
    max_seq_len = 512
    d_model = 768
    num_heads = 12
    dff = 3072
    num_layers = 12
    batch_size = 64
    dropout_rate = 0.1
    epochs = 15

    dummy_sentences = ["This is a sample sentence.", "Another sample sentence here."]
    text_processor = TextProcessor(vocab_size, max_seq_len)
    text_processor.fit_tokenizer(dummy_sentences)
    
    text_data = tf.data.TextLineDataset("large_corpus.txt").map(text_processor.preprocess_text)
    train_dataset = text_processor.prepare_dataset(text_data).shuffle(10000).batch(batch_size)
    
    model = build_model(vocab_size, d_model, num_heads, dff, num_layers, dropout_rate)
    train_model(model, train_dataset, epochs)
