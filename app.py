import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import random

# Text Corpus
text_corpus = [
    "Hello, how are you?",
    "I am fine, thank you!",
    "What are you doing today?",
    "I am learning to code.",
    "That's great! Keep it up."
]

# BPE Vocabulary Class
class BPEVocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        # Split words into characters, with special end-of-word symbol
        word_freqs = Counter(" ".join(word) + " </w>" for sentence in sentence_list for word in sentence.split())

        # Count frequency of pairs of symbols
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[symbols[i], symbols[i + 1]] += freq

        vocab = {char: idx + 4 for idx, char in enumerate(set(" ".join(word_freqs)))}
        self.stoi = {**self.stoi, **vocab}
        self.itos = {v: k for k, v in self.stoi.items()}

        # BPE Merges
        for _ in range(self.freq_threshold):
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < 2:
                break

            # Merge the most frequent pair
            new_symbol = ''.join(best_pair)
            self.stoi[new_symbol] = len(self.stoi)
            self.itos[len(self.itos)] = new_symbol

            # Update the frequencies
            new_pair_freqs = defaultdict(int)
            for pair, freq in pair_freqs.items():
                if pair == best_pair:
                    continue
                new_freq = freq
                if pair[0] == best_pair[0]:
                    new_pair = (new_symbol, pair[1])
                    new_pair_freqs[new_pair] += new_freq
                elif pair[1] == best_pair[1]:
                    new_pair = (pair[0], new_symbol)
                    new_pair_freqs[new_pair] += new_freq
                else:
                    new_pair_freqs[pair] += new_freq
            pair_freqs = new_pair_freqs

    def numericalize(self, text):
        tokenized_text = []

        # Tokenize using BPE
        for word in text.split(" "):
            word = " ".join(word) + " </w>"
            while word:
                best_match = max(self.stoi, key=lambda k: len(k) if word.startswith(k) else 0, default='<UNK>')
                tokenized_text.append(self.stoi[best_match])
                word = word[len(best_match):]

        return tokenized_text

# Next Sentence Prediction Dataset
class NSPDataset(Dataset):
    def __init__(self, text_list, vocab):
        self.text_list = text_list
        self.vocab = vocab
        self.pairs = self._create_sentence_pairs()

    def _create_sentence_pairs(self):
        pairs = []
        for i in range(len(self.text_list) - 1):
            # True pair
            true_pair = (self.text_list[i], self.text_list[i + 1], 1)
            pairs.append(true_pair)

            # False pair
            random_sentence = random.choice(self.text_list)
            false_pair = (self.text_list[i], random_sentence, 0)
            pairs.append(false_pair)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sentence1, sentence2, label = self.pairs[index]
        numericalized_text1 = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(sentence1) + [self.vocab.stoi["<EOS>"]]
        numericalized_text2 = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(sentence2) + [self.vocab.stoi["<EOS>"]]
        return torch.tensor(numericalized_text1), torch.tensor(numericalized_text2), torch.tensor(label)

# Transformer Model for NSP
class TransformerNSP(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_length=5000):
        super(TransformerNSP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Embedding(max_seq_length, embed_size) # Flexible positional encoding length
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size * 2, 2)  # Output layer for binary classification

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src1, src2):
        src1 = self.embedding(src1) + self.pos_encoder(torch.arange(0, src1.size(0)).unsqueeze(1))
        src1_mask = self.generate_square_subsequent_mask(src1.size(0))
        out1 = self.transformer_decoder(src1, src1, src1_mask)

        src2 = self.embedding(src2) + self.pos_encoder(torch.arange(0, src2.size(0)).unsqueeze(1))
        src2_mask = self.generate_square_subsequent_mask(src2.size(0))
        out2 = self.transformer_decoder(src2, src2, src2_mask)

        out = torch.cat((out1[-1], out2[-1]), dim=1)
        out = self.fc_out(out)
        return out

# Build BPE Vocabulary
vocab = BPEVocabulary(100)
vocab.build_vocabulary(text_corpus)

# Creating instances of the dataset and dataloader for NSP
nsp_dataset = NSPDataset(text_corpus, vocab)
nsp_dataloader = DataLoader(nsp_dataset, batch_size=2, shuffle=True)

# Model hyperparameters
vocab_size = len(vocab)
embed_size = 512
num_heads = 8
num_layers = 3
dropout = 0.1
max_seq_length = 5000

# Model instantiation for NSP
nsp_model = TransformerNSP(vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_length)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nsp_model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for sentence1, sentence2, label in nsp_dataloader:
        optimizer.zero_grad()
        sentence1, sentence2, label = sentence1.T, sentence2.T, label.squeeze()
        output = nsp_model(sentence1, sentence2)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Function to test NSP
def test_nsp(sentence1, sentence2):
    nsp_model.eval()
    with torch.no_grad():
        input_tensor1 = torch.LongTensor([vocab.stoi["<SOS>"]] + vocab.numericalize(sentence1) + [vocab.stoi["<EOS>"]]).unsqueeze(1)
        input_tensor2 = torch.LongTensor([vocab.stoi["<SOS>"]] + vocab.numericalize(sentence2) + [vocab.stoi["<EOS>"]]).unsqueeze(1)
        output = nsp_model(input_tensor1, input_tensor2)
        prediction = torch.argmax(output, dim=1).item()
        return "Logical Follow-up" if prediction == 1 else "Not a Logical Follow-up"

# Test example
result = test_nsp("Hello, how are you?", "I am fine, thank you!")
print(result)
