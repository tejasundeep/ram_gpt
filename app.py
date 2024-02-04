import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from collections import Counter, defaultdict
import random
import pickle
import numpy as np
import nltk
from nltk.corpus import wordnet
import spacy
import nltk.translate.bleu_score as bleu

# Download WordNet data and SpaCy's English NER model
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

# Text Corpus
text_corpus = """
A long time ago, there lived a king named Dasharatha, the ruler of the kingdom of Ayodhya. He had three wives: Kausalya, Kaikeyi, and Sumitra. The first wife, Kausalya, gave birth to the eldest son, Rama. The second wife, Kaikeyi, gave birth to Bharata, while the third wife, Sumitra, had twins and named them Lakshmana and Satrughna.
In no time, the kids grew up into handsome princes and became well-known across the kingdom for their wisdom and strength. King Dasharatha loved all his kids but had a soft spot in his heart for his eldest son, Rama.
One day, sage Vishwamitra took the young princes to the neighboring kingdom of Mithila, which King Janaka ruled. He had organized a swayamvar for his daughters, Sita and Urmila.
“Welcome, Princes!” announced King Janaka. “As you all know that I have organized this swayamvar for my two beautiful daughters, and any of you can get married to them. However, the only condition is that you have to string this great bow by Lord Shiva.”
Many princes took turns stringing the bow, but none of them succeeded. In the end, Rama went and strung it in his first attempt to win King Janaka’s elder daughter Sita for marriage. Urmila got married to Lakshmana, and all of them were welcomed back to the kingdom with great pomp and show.
Things were fine until one day, King Dasharatha expressed his willingness to throne Rama as the king.
"""

# Define hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
SEQ_LENGTH = 30

# BPE Vocabulary Class
class BPEVocabulary:
    def __init__(self, freq_threshold, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
        self.itos = {i: token for i, token in enumerate(special_tokens)}
        self.stoi = {token: i for i, token in enumerate(special_tokens)}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        words = [word.lower() for sentence in sentence_list for word in sentence.split()]
        word_freqs = Counter(words)
        vocab = {word: idx + len(self.itos) for idx, word in enumerate(word_freqs)}
        self.stoi.update(vocab)
        self.itos.update({v: k for k, v in vocab.items()})

        # BPE Merges - this can be optimized or changed depending on your BPE strategy
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
            while word:
                best_match = max([k for k in self.stoi if word.startswith(k)], key=len, default='<UNK>')
                tokens.append(self.stoi[best_match])
                word = word[len(best_match):]
        return tokens

# Text Generation Dataset with POS and NER tags
class TextGenerationDatasetWithPOSandNER(Dataset):
    def __init__(self, text_list, vocab, seq_length=30, augment_rate=0.1):
        self.vocab = vocab
        self.seq_length = seq_length
        self.augment_rate = augment_rate
        self.data = self.process_data(text_list)

    def process_data(self, text_list):
        full_text = ' '.join(text_list).lower()
        augmented_text = self.augment_sentence(full_text)
        words = nltk.word_tokenize(augmented_text)
        pos_tags = nltk.pos_tag(words)
        ner_tags = [ent.label_ for ent in nlp(augmented_text).ents]
        
        tokens = self.vocab.numericalize(words)
        pos_indices = [self.vocab.stoi.get(tag, self.vocab.stoi['<UNK>']) for (_, tag) in pos_tags]
        ner_indices = [self.vocab.stoi.get(tag, self.vocab.stoi['<UNK>']) for tag in ner_tags]
        
        # Combine word, POS, and NER tag indices
        combined_indices = [f"{token}_{pos}_{ner}" for token, pos, ner in zip(tokens, pos_indices, ner_indices)]
        
        sequences = [combined_indices[i:i + self.seq_length] for i in range(len(tokens) - self.seq_length)]
        return sequences

    def augment_sentence(self, sentence):
        words = sentence.split()
        new_words = words.copy()
        n_aug = int(np.ceil(self.augment_rate * len(words)))
        augmented_indices = np.random.choice(len(words), n_aug, replace=False)

        for i in augmented_indices:
            action = np.random.choice(["synonym", "insert", "delete", "swap"])
            if action == "synonym":
                synonyms = self.get_synonyms(words[i])
                if len(synonyms) > 0:
                    new_words[i] = np.random.choice(synonyms)
            elif action == "insert":
                synonyms = self.get_synonyms(words[i])
                if len(synonyms) > 0:
                    new_words.insert(i, np.random.choice(synonyms))
            elif action == "delete":
                new_words.pop(i)
            elif action == "swap":
                swap_idx = np.random.choice(len(words))
                new_words[i], new_words[swap_idx] = new_words[swap_idx], new_words[i]
        return ' '.join(new_words)

    @staticmethod
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').replace('-', ' ').lower()
                synonym = "".join([char for char in synonym if char.isalpha()])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# Transformer Text Generation Model (Modified for Dynamic Sequence Length)
class TransformerTextGeneratorWithLinguisticFeatures(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout, max_seq_length=5000):
        super(TransformerTextGeneratorWithLinguisticFeatures, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Embedding(max_seq_length, embed_size)
        self.ner_encoder = nn.Embedding(vocab_size, embed_size)  # Embedding layer for NER tags
        self.fc_pos = nn.Linear(embed_size, embed_size)  # Fully connected layer for POS tags
        self.fc_ner = nn.Linear(embed_size, embed_size)  # Fully connected layer for NER tags
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Add multihead attention layers
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.multihead_attention_pos = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.multihead_attention_ner = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)

    def forward(self, input_seq, max_length=50, pos_tags=None, ner_tags=None):
        batch_size, seq_length = input_seq.size()
        src = self.embedding(input_seq) + self.pos_encoder(torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1))
        src = self.layer_norm(src)

        # Apply multihead attention to the input sequence
        src2 = src.permute(1, 0, 2)  # Transpose for multihead attention
        src2, _ = self.multihead_attention(src2, src2, src2)
        src2 = src2.permute(1, 0, 2)  # Transpose back to (batch_size, seq_length, embed_size)

        # Embed POS tags and apply multihead attention
        pos_tags = self.fc_pos(self.embedding(pos_tags))  # Apply a linear layer to POS embeddings
        pos_tags = pos_tags.permute(1, 0, 2)  # Transpose for multihead attention
        src2, _ = self.multihead_attention_pos(src2, pos_tags, pos_tags)
        src2 = src2.permute(1, 0, 2)  # Transpose back to (batch_size, seq_length, embed_size)

        # Embed NER tags and apply multihead attention
        ner_tags = self.fc_ner(self.embedding(ner_tags))  # Apply a linear layer to NER embeddings
        ner_tags = ner_tags.permute(1, 0, 2)  # Transpose for multihead attention
        src2, _ = self.multihead_attention_ner(src2, ner_tags, ner_tags)
        src2 = src2.permute(1, 0, 2)  # Transpose back to (batch_size, seq_length, embed_size)

        generated_sequence = []

        for i in range(max_length):
            tgt_mask = self.generate_autoregressive_mask(seq_length + i).to(src.device)
            output = self.transformer_decoder(src2, src2, tgt_mask=tgt_mask)

            last_output = output[:, -1, :]
            token_probs = self.fc_out(last_output)
            next_token = token_probs.argmax(dim=-1)

            generated_sequence.append(next_token.unsqueeze(1))
            src2 = torch.cat([src2, self.embedding(next_token)], dim=1)

        generated_sequence = torch.cat(generated_sequence, dim=1)
        return generated_sequence

    @staticmethod
    def generate_autoregressive_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

# Model and Dataset Initialization
vocab = BPEVocabulary(100)
vocab.build_vocabulary([text_corpus])
text_gen_dataset = TextGenerationDatasetWithPOSandNER([text_corpus], vocab, SEQ_LENGTH)

train_size = int(0.8 * len(text_gen_dataset))
val_size = len(text_gen_dataset) - train_size
train_dataset, val_dataset = random_split(text_gen_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Parameters
params = {'vocab_size': len(vocab), 'embed_size': 512, 'num_heads': 8, 'num_layers': 3, 'dropout': 0.1, 'max_seq_length': 5000}
text_gen_model = TransformerTextGeneratorWithLinguisticFeatures(**params)

# Training Setup
mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
optimizer = optim.Adam(text_gen_model.parameters(), lr=0.001)

# Define a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

accumulation_steps = 4  # You can adjust this value based on your needs

for epoch in range(NUM_EPOCHS):
    text_gen_model.train()
    total_mlm_loss = 0
    total_bleu_reward = 0

    for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward pass to generate text
        output = text_gen_model(input_seq.T)
        output = output.view(-1, output.size(-1))

        # Calculate the MLM loss
        mlm_loss = mlm_loss_fn(output, target_seq.T.contiguous().view(-1))

        # Calculate BLEU score as a reward
        reference = [target_seq.tolist()]  # Convert to list of lists
        candidate = output.argmax(dim=-1).tolist()
        bleu_score = bleu.sentence_bleu(reference, candidate)

        # Introduce a reward term based on BLEU score
        reward = torch.tensor(bleu_score, dtype=torch.float32, device=output.device)
        total_bleu_reward += reward.item()

        # Compute the reward loss
        reward_loss = -reward  # You can customize this based on your specific criteria

        # Combine the MLM loss and reward loss
        combined_loss = mlm_loss + reward_loss

        # Backward pass
        combined_loss.backward()

        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
            # Clip gradients to prevent exploding gradients
            max_grad_norm = 1.0  # You can adjust this value as needed
            utils.clip_grad_norm_(text_gen_model.parameters(), max_grad_norm)

            # Update parameters
            optimizer.step()

            # Clear accumulated gradients
            optimizer.zero_grad()

        total_mlm_loss += mlm_loss.item()

    # Step the learning rate scheduler
    scheduler.step()

    text_gen_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_dataloader:
            output = text_gen_model(input_seq.T)
            output = output.view(-1, output.size(-1))
            target_seq = target_seq.T.contiguous().view(-1)
            val_loss = mlm_loss_fn(output, target_seq)
            total_val_loss += val_loss.item()

    avg_mlm_loss = total_mlm_loss / len(train_dataloader)
    avg_bleu_reward = total_bleu_reward / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch}: Avg. MLM Loss = {avg_mlm_loss:.4f}, Avg. BLEU Reward = {avg_bleu_reward:.4f}, Avg. Val Loss = {avg_val_loss:.4f}, Learning Rate = {scheduler.get_lr()[0]}")

# Save model and vocabulary
torch.save(text_gen_model.state_dict(), 'text_gen_model.pth')
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Text Generation Function with POS and NER tags
def generate_text_with_linguistic_features(model, start_seq, vocab, max_length=50, p=0.95, pos_tags=None, ner_tags=None):
    model.eval()
    words = start_seq.lower().split()
    input_seq = torch.tensor([[vocab.stoi.get(word, vocab.stoi['<UNK>']) for word in words]], dtype=torch.long)
    pos_tags = [vocab.stoi.get(tag, vocab.stoi['<UNK>']) for tag in pos_tags]  # Convert POS tags to indices
    ner_tags = [vocab.stoi.get(tag, vocab.stoi['<UNK>']) for tag in ner_tags]  # Convert NER tags to indices
    
    for _ in range(max_length):
        output = model(input_seq.T, pos_tags=torch.tensor([pos_tags]), ner_tags=torch.tensor([ner_tags]))
        token_probs = F.softmax(output[-1, :], dim=-1)  # Apply softmax to the last token's output

        # Sort the probabilities and find the smallest set of tokens whose cumulative probability exceeds p
        sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices = sorted_indices[cumulative_probs <= p]

        # Randomly sample the next token from this set
        next_token_idx = random.choices(sorted_indices.tolist(), weights=sorted_probs[cumulative_probs <= p].tolist())[0]
        next_word = vocab.itos[next_token_idx]

        words.append(next_word)
        if next_word == '<EOS>':
            break
        input_seq = torch.cat([input_seq, torch.tensor([[next_token_idx]], dtype=torch.long)], dim=1)

    return ' '.join(words)

# Example usage with POS and NER tags
start_seq = "Who is Dasharatha?"
pos_tags = ["WP", "VBZ", "NNP", "NNP"]
ner_tags = ["O", "O", "PERSON", "PERSON"]

generated_text = generate_text_with_linguistic_features(text_gen_model, start_seq, vocab, p=0.95, pos_tags=pos_tags, ner_tags=ner_tags)
print(generated_text)
