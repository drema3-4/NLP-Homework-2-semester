import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RoPe(nn.Module):
    def __init__(self, max_context_length, d_head, base = 10000.0):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_head = d_head
        self.base = base

        self.__build_rope_cache__()
    
    def __build_rope_cache__(self):
        positions = torch.arange(self.max_context_length)
        pair_idx = torch.arange(self.d_head // 2)

        theta = self.base ** (-2.0 * pair_idx / self.d_head)

        angles = positions[:, None] * theta[None, :]
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

        self.register_buffer("cos", torch.cos(angles).unsqueeze(0), persistent=False)
        self.register_buffer("sin", torch.sin(angles).unsqueeze(0), persistent=False)       

    def __rotate_half__(self, x):
        x_odd = x[..., 0::2]
        x_even = x[..., 1::2]

        rotate_half_x = torch.stack((-x_even, x_odd), dim=-1)

        return rotate_half_x.flatten(start_dim=-2)
    
    def forward(self, x):
        seq_len = x.size(-2)

        cos = self.cos[:, :seq_len, :]
        sin = self.sin[:, :seq_len, :]

        return x * cos + self.__rotate_half__(x) * sin
    

class SelfAttention(nn.Module):
    def __init__(self, max_context_length, d_model, d_head, base=10000.0, dropout=0.1):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model
        self.d_head = d_head
        self.base = base
        self.dropout = dropout

        mean = 0.0
        std = math.sqrt(2.0 / (self.d_model + self.d_head))

        self.W_Q = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.W_K = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.RoPe = RoPe(self.max_context_length, self.d_head, self.base)
        self.W_V = nn.Parameter(torch.empty(d_model, d_head).normal_(mean, std))
        self.dropoutA = nn.Dropout(self.dropout)

    def forward(self, X):
        Q = self.RoPe(X @ self.W_Q)
        K = self.RoPe(X @ self.W_K)
        V = X @ self.W_V

        S = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head)

        causal_mask = ~torch.tril(torch.ones_like(S, dtype=torch.bool))
        S = S.masked_fill(causal_mask, -torch.inf)
        S = torch.softmax(S, dim=-1)

        A = S @ V

        return self.dropoutA(A)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, max_context_length, d_model, d_head, num_heads, base=10000.0, dropout=0.1):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.base = base
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [SelfAttention(self.max_context_length, self.d_model, self.d_head, self.base, self.dropout) for _ in range(self.num_heads)]
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
    

class Expert(nn.Module):
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
    

class MoE(nn.Module):
    def __init__(self, num_experts, top_k, d_model, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.dropout = dropout

        self.router = nn.Linear(self.d_model, self.num_experts)
        
        self.experts = nn.ModuleList([Expert(self.d_model, self.dropout) for _ in range(self.num_experts)])

    def forward(self, X):
        if len(X.size()) == 3:
            B, T, D = X.size()
            X_flatten = X.flatten(start_dim=0, end_dim=1)
        else:
            X_flatten = X

        experts_weights = F.softmax(self.router(X_flatten), dim=-1)
        experts_importance = experts_weights.mean(dim=0)
        experts_weights_norms = experts_weights.topk(self.top_k, dim=-1).values.sum(dim=-1)
        experts_ids = experts_weights.topk(self.top_k, dim=-1).indices

        output_flatten = torch.zeros_like(X_flatten)
        for expert_number, expert in enumerate(self.experts):
            mask = (experts_ids == expert_number).any(dim=1)
            if not mask.any().item():
                continue

            expert_input = X_flatten[mask]
            expert_output = expert(expert_input)

            output_flatten[mask] += (experts_weights[mask, expert_number] / experts_weights_norms[mask]).unsqueeze(-1) * expert_output      

        if len(X.size()) == 3:
            output_flatten = output_flatten.view(B, T, D)

        u = torch.ones_like(experts_importance) / self.num_experts
        aux_loss = torch.mean((experts_importance - u)**2)

        return output_flatten, aux_loss


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
    def __init__(self, max_context_length, d_model, num_heads, d_head, num_experts, top_k, base=10000.0, dropout=0.1, epsilon=1e-6):
        super().__init__()
        self.max_context_length = max_context_length
        self.d_model = d_model if d_model % num_heads == 0 else d_model // num_heads * num_heads
        self.num_heads = num_heads
        self.d_head = d_head
        self.num_experts = num_experts
        self.top_k = top_k
        self.base = base
        self.dropout = dropout
        self.epsilon = epsilon

        self.layerNormPreMHA = LayerNorm(self.d_model, self.epsilon)
        self.multiHeadAttention = MultiHeadAttention(
            self.max_context_length,
            self.d_model,
            self.d_head,
            self.num_heads,
            self.base,
            self.dropout
        )
        self.layerNormPreFF = LayerNorm(self.d_model, self.epsilon)
        self.MoE = MoE(self.num_experts, self.top_k, self.d_model, self.dropout)

    def forward(self, embeddings):
        embeddings = embeddings + self.multiHeadAttention(self.layerNormPreMHA(embeddings))
        empbeddings_after_moe, loss = self.MoE(self.layerNormPreFF(embeddings))
        embeddings = embeddings + empbeddings_after_moe

        return embeddings, loss
    

class GPTPreNormWithRoPeWithMoE(nn.Module):
    def __init__(self, max_context_length, num_encoder_blocks, vocab_size, d_model, num_heads, num_experts, top_k, base=10000.0, dropout=0.1):
        super().__init__()
        self.max_context_length = max_context_length
        self.num_encoder_blocks = num_encoder_blocks
        self.vocab_size = vocab_size
        self.d_model = d_model if d_model % num_heads == 0 else d_model // num_heads * num_heads
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.base = base
        self.dropout = dropout
        self.epsilon = 1e-6

        self.embeddingLayer = EmbeddingLayer(self.vocab_size, self.d_model)

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    self.max_context_length,
                    self.d_model,
                    self.num_heads,
                    self.d_head,
                    self.num_experts,
                    self.top_k,
                    self.base,
                    self.dropout,
                    self.epsilon
                ) for _ in range(self.num_encoder_blocks)
            ]
        )

        self.finalLayerNorm = LayerNorm(self.d_model, self.epsilon)
        self.embeddings2ids = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, batch_token_ids):
        embeddings = self.embeddingLayer(batch_token_ids)

        embeddings_after_encoder = embeddings
        aux_loss = 0.0
        for encoder in self.encoders:
            embeddings_after_encoder, encoder_aux_loss = encoder(embeddings_after_encoder)
            aux_loss = aux_loss + encoder_aux_loss

        aux_loss = aux_loss / self.num_encoder_blocks

        embeddings_after_encoder = self.finalLayerNorm(embeddings_after_encoder)
        gen_batch_token_ids = self.embeddings2ids(embeddings_after_encoder)

        return gen_batch_token_ids, aux_loss

    def generate(self, start_token_ids, num_gen_tokens):
        squeeze_output = start_token_ids.dim() == 1
        gen_token_ids = start_token_ids.unsqueeze(0) if squeeze_output else start_token_ids

        for _ in range(num_gen_tokens):
            context = gen_token_ids[:, -self.max_context_length:]

            embeddings = self.embeddingLayer(context)

            embeddings_after_encoder = embeddings
            aux_loss = 0.0
            for encoder in self.encoders:
                embeddings_after_encoder, encoder_aux_loss = encoder(embeddings_after_encoder)
                aux_loss = aux_loss + encoder_aux_loss

            aux_loss = aux_loss / self.num_encoder_blocks

            embeddings_after_encoder = self.finalLayerNorm(embeddings_after_encoder)
            gen_token_probabilities = F.softmax(self.embeddings2ids(embeddings_after_encoder)[:, -1], dim=-1)
            gen_token_id = torch.multinomial(gen_token_probabilities, num_samples=1)

            gen_token_ids = torch.concat([gen_token_ids, gen_token_id], dim=-1)

        return gen_token_ids.squeeze(0) if squeeze_output else gen_token_ids