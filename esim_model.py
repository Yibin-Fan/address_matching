import torch
from torch import nn


class ESIM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word2vec_matrix, max_sequence_length, hidden_dim=128):
        super(ESIM_Model, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec_matrix))
        self.embedding.weight.requires_grad = False

        # BiLSTM encoder
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Local inference modeling
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Inference composition
        self.composition = nn.LSTM(hidden_dim * 8, hidden_dim, batch_first=True, bidirectional=True)

        # Output layer
        self.classification = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().unsqueeze(-1)
        mask2 = mask2.float().unsqueeze(1)
        attention = attention * mask1 * mask2

        # Softmax attention weights
        weight1 = torch.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = torch.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        return x1_align, x2_align

    def forward(self, premise, hypothesis):
        # Create masks
        premise_mask = premise != 0
        hypothesis_mask = hypothesis != 0

        # Embedding
        premise_embed = self.embedding(premise)
        hypothesis_embed = self.embedding(hypothesis)

        # BiLSTM encoding
        premise_encoded, _ = self.lstm(premise_embed)
        hypothesis_encoded, _ = self.lstm(hypothesis_embed)

        # Local inference
        premise_align, hypothesis_align = self.soft_attention_align(
            premise_encoded, hypothesis_encoded,
            premise_mask, hypothesis_mask
        )

        # Enhancement of local inference information
        premise_enhanced = torch.cat([
            premise_encoded,
            premise_align,
            premise_encoded - premise_align,
            premise_encoded * premise_align
        ], dim=-1)
        hypothesis_enhanced = torch.cat([
            hypothesis_encoded,
            hypothesis_align,
            hypothesis_encoded - hypothesis_align,
            hypothesis_encoded * hypothesis_align
        ], dim=-1)

        # Inference composition
        premise_composed, _ = self.composition(premise_enhanced)
        hypothesis_composed, _ = self.composition(hypothesis_enhanced)

        # Pooling
        premise_avg_pool = torch.mean(premise_composed, dim=1)
        premise_max_pool = torch.max(premise_composed, dim=1)[0]
        hypothesis_avg_pool = torch.mean(hypothesis_composed, dim=1)
        hypothesis_max_pool = torch.max(hypothesis_composed, dim=1)[0]

        # Classification
        pooled = torch.cat([
            premise_avg_pool,
            premise_max_pool,
            hypothesis_avg_pool,
            hypothesis_max_pool
        ], dim=-1)

        return self.classification(pooled)