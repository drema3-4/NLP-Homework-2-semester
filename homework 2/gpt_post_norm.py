import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        mean = 0.0
        std = math.sqrt(2.0 / (self.d_model + self.d_head))

        self.W_Q = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.W_K = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.W_V = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.dropoutA = nn.Dropout(self.dropout)

    def forward(self, X):
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head)

        causal_mask = ~torch.tril(torch.ones_like(S, dtype=torch.bool))
        S = S.masked_fill(causal_mask, -torch.inf)
        S = torch.softmax(S, dim=-1)

        A = S @ V

        return self.dropoutA(A)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [SelfAttention(self.d_model, self.d_head, self.dropout) for _ in range(self.num_heads)]
        )
        self.proj = nn.Linear(self.d_model, self.d_model)
        self.dropoutMHA = nn.Dropout(self.dropout)

    def forward(self, X):
        X = torch.concat([head(X) for head in self.heads], dim=-1)
        X = self.proj(X)

        return self.dropoutMHA(X)
    

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        mean = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, unbiased=False, keepdim=True)

        X = ((X - mean) / torch.sqrt(var + self.epsilon)) * self.gamma + self.beta

        return X
    

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.layer1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(4 * self.d_model, self.d_model)
        self.dropoutFF = nn.Dropout(self.dropout)

    def forward(self, X):
        X = self.layer2(
            self.act(
                self.layer1(X)
            )
        )

        return self.dropoutFF(X)
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_context_length, d_model):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model

        mean = 0.0
        std = math.sqrt(1.0 / self.d_model)

        self.position_codes = nn.Parameter(torch.empty(self.max_context_length, self.d_model).normal_(mean, std))

    def forward(self, seq_length):
        return self.position_codes[:seq_length]
    

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        mean = 0.0
        std = math.sqrt(1.0 / self.d_model)

        self.embeddings = nn.Parameter(torch.empty(self.vocab_size, self.d_model).normal_(mean, std))

    def forward(self, token_ids):
        return self.embeddings[token_ids]
    

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_head, dropout=0.1, epsilon=1e-6):
        super().__init__()
        self.d_model = d_model if d_model % num_heads == 0 else d_model // num_heads * num_heads
        self.num_heads = num_heads
        self.d_head = d_head
        self.dropout = dropout
        self.epsilon = epsilon

        self.multiHeadAttention = MultiHeadAttention(self.d_model, self.d_head, self.num_heads, self.dropout)
        self.layerNormMHA = LayerNorm(self.d_model, self.epsilon)
        self.feedForward = FeedForward(self.d_model, self.dropout)
        self.layerNormFF = LayerNorm(self.d_model, self.epsilon)

    def forward(self, embeddings):
        embeddings = self.layerNormMHA(embeddings + self.multiHeadAttention(embeddings))
        embeddings = self.layerNormFF(embeddings + self.feedForward(embeddings))

        return embeddings
    

class GPTPostNorm(nn.Module):
    def __init__(self, max_context_length, num_encoder_blocks, vocab_size, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.max_context_length = max_context_length
        self.num_encoder_blocks = num_encoder_blocks
        self.vocab_size = vocab_size
        self.d_model = d_model if d_model % num_heads == 0 else d_model // num_heads * num_heads
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads
        self.dropout = dropout
        self.epsilon = 1e-6

        self.embeddingLayer = EmbeddingLayer(self.vocab_size, self.d_model)
        self.positionalEnconding = PositionalEncoding(self.max_context_length, self.d_model)

        self.encoders = nn.ModuleList(
            [Encoder(self.d_model, self.num_heads, self.d_head, self.dropout, self.epsilon) for _ in range(self.num_encoder_blocks)]
        )

        self.finalLayerNorm = LayerNorm(self.d_model, self.epsilon)
        self.embeddings2ids = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, batch_token_ids):
        seq_length = batch_token_ids.size(1)
        embeddings = self.embeddingLayer(batch_token_ids) + self.positionalEnconding(seq_length)

        encoder_outputs = embeddings
        for encoder in self.encoders:
            encoder_outputs = encoder(encoder_outputs)

        encoder_outputs = self.finalLayerNorm(encoder_outputs)
        gen_batch_token_ids = self.embeddings2ids(encoder_outputs)

        return gen_batch_token_ids

    def generate(self, start_token_ids, num_gen_tokens):
        squeeze_output = start_token_ids.dim() == 1
        gen_token_ids = start_token_ids.unsqueeze(0) if squeeze_output else start_token_ids

        for _ in range(num_gen_tokens):
            context = gen_token_ids[:, -self.max_context_length:]
            seq_length = context.size(1)

            embeddings = self.embeddingLayer(context) + self.positionalEnconding(seq_length)

            encoder_outputs = embeddings
            for encoder in self.encoders:
                encoder_outputs = encoder(encoder_outputs)

            encoder_outputs = self.finalLayerNorm(encoder_outputs)
            gen_token_probabilities = F.softmax(self.embeddings2ids(encoder_outputs)[:, -1], dim=-1)
            gen_token_id = torch.multinomial(gen_token_probabilities, num_samples=1)

            gen_token_ids = torch.concat([gen_token_ids, gen_token_id], dim=-1)

        return gen_token_ids.squeeze(0) if squeeze_output else gen_token_ids