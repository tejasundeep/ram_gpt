import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter, defaultdict
import random

# Text Corpus
text_corpus = [
    "Hello, how are you?", "I am fine, thank you!",
    "What are you doing today?", "I am learning to code.",
    "That's great! Keep it up."
]

# BPE Vocabulary Class
class BPEVocabulary:
    def __init__(self, freq_threshold, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
        self.itos = {i: token for i, token in enumerate(special_tokens)}
        self.stoi = {token: i for i, token in enumerate(special_tokens)}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        words = [word for sentence in sentence_list for word in sentence.split()]
        word_freqs = Counter(" ".join(word) + " </w>" for word in words)
        symbols = set(" ".join(word_freqs))
        vocab = {char: idx + len(self.itos) for idx, char in enumerate(symbols)}
        self.stoi.update(vocab)
        self.itos.update({v: k for k, v in vocab.items()})

        # BPE Merges
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freqs[symbols[i], symbols[i + 1]] += freq

        for _ in range(self.freq_threshold):
            if not pair_freqs or max(pair_freqs.values()) < 2:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_symbol = ''.join(best_pair)
            self.stoi[new_symbol] = len(self.stoi)
            self.itos[len(self.itos)] = new_symbol

            new_pair_freqs = defaultdict(int)
            for pair, freq in pair_freqs.items():
                if pair != best_pair:
                    new_pair = (pair[0] if pair[0] != best_pair[0] else new_symbol,
                                pair[1] if pair[1] != best_pair[1] else new_symbol)
                    new_pair_freqs[new_pair] += freq
            pair_freqs = new_pair_freqs

    def numericalize(self, text):
        tokens = []
        for word in text.split():
            word = " ".join(word) + " </w>"
            while word:
                best_match = max([k for k in self.stoi.keys() if word.startswith(k)], key=len, default='<UNK>')
                tokens.append(self.stoi[best_match])
                word = word[len(best_match):]
        return tokens

# Next Sentence Prediction Dataset
class NSPDataset(Dataset):
    def __init__(self, text_list, vocab):
        self.text_list = text_list
        self.vocab = vocab
        self.pairs = []
        for i in range(len(self.text_list) - 1):
            self.pairs.append((self.text_list[i], self.text_list[i + 1], 1))  # Positive pair
            neg_pair = random.choice([t for t in self.text_list if t != self.text_list[i + 1]])
            self.pairs.append((self.text_list[i], neg_pair, 0))  # Negative pair

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
        self.pos_encoder = nn.Embedding(max_seq_length, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_size * 2, 2)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz), 1).T.masked_fill(torch.triu(torch.ones(sz, sz), 1) == 1, float('-inf'))

    def forward(self, src1, src2):
        src1 = self.embedding(src1) + self.pos_encoder(torch.arange(src1.size(0)).unsqueeze(1))
        src2 = self.embedding(src2) + self.pos_encoder(torch.arange(src2.size(0)).unsqueeze(1))

        mask = self.generate_square_subsequent_mask(src1.size(0))

        out1 = self.transformer_encoder(src1, mask=mask)
        out2 = self.transformer_encoder(src2, mask=mask)

        out = self.fc_out(torch.cat((out1[-1], out2[-1]), dim=1))
        return out

# Model and Dataset Initialization
vocab = BPEVocabulary(100)
vocab.build_vocabulary(text_corpus)
nsp_dataset = NSPDataset(text_corpus, vocab)

# Splitting dataset into training and validation
train_size = int(0.8 * len(nsp_dataset))
val_size = len(nsp_dataset) - train_size
train_dataset, val_dataset = random_split(nsp_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model Parameters
params = {'vocab_size': len(vocab), 'embed_size': 512, 'num_heads': 8, 'num_layers': 3, 'dropout': 0.1, 'max_seq_length': 5000}
nsp_model = TransformerNSP(**params)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nsp_model.parameters(), lr=0.001)
num_epochs = 10

# Training and Validation Loop
for epoch in range(num_epochs):
    nsp_model.train()
    for sentence1, sentence2, label in train_dataloader:
        optimizer.zero_grad()
        output = nsp_model(sentence1.T, sentence2.T)
        loss = criterion(output, label.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nsp_model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation phase
    nsp_model.eval()
    total_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for sentence1, sentence2, label in val_dataloader:
            output = nsp_model(sentence1.T, sentence2.T)
            loss = criterion(output, label.squeeze())
            total_loss += loss.item()
            total_accuracy += (output.argmax(1) == label).sum().item()

    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = total_accuracy / len(val_dataloader.dataset)
    print(f"Epoch {epoch}: Avg. Loss = {avg_loss:.4f}, Avg. Accuracy = {avg_accuracy:.4f}")

# Test Function
def test_nsp(sentence1, sentence2, vocab, model):
    model.eval()
    with torch.no_grad():
        input_tensor1 = torch.LongTensor([vocab.stoi["<SOS>"]] + vocab.numericalize(sentence1) + [vocab.stoi["<EOS>"]]).unsqueeze(1)
        input_tensor2 = torch.LongTensor([vocab.stoi["<SOS>"]] + vocab.numericalize(sentence2) + [vocab.stoi["<EOS>"]]).unsqueeze(1)
        output = model(input_tensor1, input_tensor2)
        return "Logical Follow-up" if torch.argmax(output, dim=1).item() == 1 else "Not a Logical Follow-up"

# Test Example
print(test_nsp("Hello, how are you?", "I am fine, thank you!", vocab, nsp_model))
