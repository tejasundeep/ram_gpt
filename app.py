import json
import random
import re
from pathlib import Path

import numpy as np
import rasa.nlu
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 max_position_encoding, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding,
                               dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding,
                               dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        encoder_inputs, decoder_inputs = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(encoder_inputs, decoder_inputs)

        enc_output = self.encoder(encoder_inputs, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(decoder_inputs, enc_output, training, look_ahead_mask,
                                                     dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    @staticmethod
    def create_masks(encoder_inputs, decoder_inputs):
        enc_padding_mask = tf.cast(tf.math.equal(encoder_inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        dec_padding_mask = tf.cast(tf.math.equal(encoder_inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

        look_ahead_mask = Transformer.create_look_ahead_mask(tf.shape(decoder_inputs)[1])
        dec_target_padding_mask = tf.cast(tf.math.equal(decoder_inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.positional_encoding = self.positional_encoding(max_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

    @staticmethod
    def positional_encoding(position, d_model):
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    @staticmethod
    def point_wise_feed_forward_network(d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_position_encoding,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.positional_encoding = self.positional_encoding(max_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, block1, block2

class Chatbot:
    def __init__(self):
        self.model_filename = 'trained_transformer_model.h5'
        self.trained_files_filename = 'trained_files.json'
        self.nlu_model_path = 'path_to_nlu_model_directory'
        self.conversation_history = []
        self.sentences = None
        self.load_trained_files()
        self.load_transformer_model()

    def load_trained_files(self):
        trained_files_path = Path(self.trained_files_filename)
        if trained_files_path.exists():
            with trained_files_path.open('r') as file:
                self.trained_files = json.load(file)
        else:
            self.trained_files = []

    def read_data(self, directory):
        directory_path = Path(directory)
        files = [
            file
            for file in directory_path.iterdir()
            if file.suffix == '.txt' and str(file) not in self.trained_files
        ]

        contents = []
        for file in files:
            try:
                with file.open('r') as f:
                    original_sentence = re.sub(r'\s+', ' ', f.read()).strip()
                    augmented_sentence = self.random_swap(original_sentence)
                    contents.extend([original_sentence, augmented_sentence])
                    self.trained_files.append(str(file))
            except IOError:
                print(f'Error opening {file}')
        return contents

    @staticmethod
    def random_swap(sentence, n=5):
        sentence = sentence.split()
        length = len(sentence)
        n = min(n, length)
        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
        return ' '.join(sentence)

    @staticmethod
    def nucleus_sampling(input_seq, output_seq, transformer_model, p, word_index, max_length=50):
        end_token = word_index["<EOS>"]
        for _ in tf.range(max_length):
            padded_output = pad_sequences([output_seq], maxlen=input_seq.shape[1] - 1, padding='pre')
            predictions = transformer_model([input_seq, padded_output], training=False)
            probs = predictions[0, -1]
            idx = tf.squeeze(tf.where(tf.cumsum(probs) > p), axis=-1)
            if tf.size(idx) == 0:
                idx = [tf.argmax(probs)]
            sampled_token = tf.random.categorical(tf.math.log(tf.expand_dims(probs[idx], axis=0)), num_samples=1)[0, 0]
            if sampled_token == end_token or tf.size(output_seq) >= max_length:
                break
            output_seq.append(sampled_token)
        return output_seq

    def preprocess_data(self, dir_path):
        self.sentences = [self.random_swap(sentence) for sentence in self.read_data(dir_path)]
        if not self.sentences:
            raise ValueError("No sentences found for preprocessing. Please check the directory and trained_files.")

        self.tokenizer = Tokenizer(filters="\"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n", lower=True, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.sentences)
        self.word_index = self.tokenizer.word_index
        self.word_index["<SOS>"], self.word_index["<EOS>"] = len(self.word_index) + 1, len(self.word_index) + 1
        input_sequences = pad_sequences(self.tokenizer.texts_to_sequences(self.sentences), padding='pre')
        input_data, target_data = input_sequences[:, :-1], np.zeros_like(input_sequences)
        target_data[:, :-1], target_data[:, -1] = input_sequences[:, 1:], self.word_index["<EOS>"]
        return input_data, target_data

    def load_transformer_model(self):
        model_path = Path(self.model_filename)
        if model_path.exists():
            self.transformer_model = tf.keras.models.load_model(self.model_filename, compile=False)

    def save_trained_files(self):
        with Path(self.trained_files_filename).open('w') as file:
            json.dump(self.trained_files, file)

    def train_transformer_model(self, input_data, target_data):
        optimizer = tf.keras.optimizers.Adam(0.001)

        def custom_loss(y_true, y_pred):
            mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=y_pred.dtype)
            loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true, y_pred) * mask) / tf.reduce_sum(mask)
            return loss

        if self.sentences:
            if not self.transformer_model:
                self.transformer_model = Transformer(
                    num_layers=4,
                    d_model=128,
                    num_heads=4,
                    dff=512,
                    input_vocab_size=len(self.word_index) + 1,
                    target_vocab_size=len(self.word_index) + 1,
                    max_position_encoding=input_data.shape[1] - 1,
                    dropout_rate=0.1
                )

            self.transformer_model.compile(optimizer=optimizer, loss=custom_loss, metrics=["accuracy"])
            self.transformer_model.fit(input_data, target_data, batch_size=64, epochs=10, validation_split=0.1)
            self.transformer_model.save(self.model_filename)

    def load_nlu_model(self):
        self.nlu_model = rasa.nlu.load(self.nlu_model_path)

    def generate_tree(self):
        root = Node("Conversation")
        current_node = root
        for sentence in self.conversation_history:
            if sentence.startswith("You: "):
                node = Node(sentence[4:], parent=current_node)
            else:
                node = Node(sentence[8:], parent=current_node)
            current_node = node
        return root

    def process_input_text(self, input_text):
        input_text_engineered = input_text + " Please think step by step before answering."
        self.conversation_history.append("You: " + input_text_engineered)
        intent = self.nlu_model.parse(input_text)["intent"]["name"]
        self.conversation_history.append(f"Intent: {intent}")

    def generate_response(self):
        input_seq = pad_sequences(
            self.tokenizer.texts_to_sequences([" ".join(self.conversation_history[:-1])]),
            maxlen=input_data.shape[1] - 1,
            padding='pre'
        )
        generated_sequence = self.nucleus_sampling(
            input_seq,
            [self.word_index["<SOS>"]],
            self.transformer_model,
            0.9,
            self.word_index,
            50
        )
        response = self.tokenizer.sequences_to_texts([generated_sequence[1:]])[0]
        self.conversation_history.append("Chatbot: " + response)
        return response

    def run(self, dir_path):
        try:
            input_data, target_data = self.preprocess_data(dir_path)
            self.train_transformer_model(input_data, target_data)
            self.load_nlu_model()

            while True:
                try:
                    input_text = input("You: ")
                    if input_text.lower() in ["bye", "goodbye", "exit"]:
                        break

                    self.process_input_text(input_text)

                    response = self.generate_response()
                    print("Chatbot:", response)

                    tree = self.generate_tree()
                    print("Tree of Thoughts:")
                    for pre, _, node in RenderTree(tree):
                        print("%s%s" % (pre, node.name))

                except KeyboardInterrupt:
                    print("Conversation ended by user.")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

            self.save_trained_files()

        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except ValueError as e:
            print(f"Error: {str(e)}")


if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.run("your_ebook_directory_path")
