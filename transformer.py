import torch
from torch import nn
import torch.nn.functional as F
import math

d_model = 512
d_vocab = 10000
max_tokens = 1000

class AttentionHead(nn.Module):
    def __init__(self, d_k, d_v):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_k)
        self.w_k = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_v)
        self.sqrt = math.sqrt(d_k)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        x = torch.matmul(q, k.t()) / self.sqrt
        mask = torch.triu(torch.ones_like(x), diagonal=1)
        x = F.softmax(x - 10**10 * mask, dim=1)
        x = torch.matmul(x, v)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_k, d_v):
        super().__init__()
        self.heads = [AttentionHead(d_k=d_k, d_v=d_v) for _ in range(h)]
        self.w_o = nn.Linear(h*d_v, d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]     # This can be parallelized
        x = torch.cat(head_outputs, dim=1)
        x = self.w_o(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, p_drop, d_ff, h, d_k, d_v):
        super().__init__()
        self.attention = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

    def forward(self, x):
        x = self.layer_norm1(x + self.dropout1(self.attention(x)))
        x = self.layer_norm2(x + self.dropout2(self.ff(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers=6, p_drop=0.1, d_ff=2048, h=8, d_k=64, d_v=64):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DecoderBlock(p_drop=p_drop, d_ff=d_ff, h=h, d_k=d_k, d_v=d_v))
        layers.append(nn.Linear(d_model, d_vocab))
        self.stack = nn.Sequential(*layers)
        #self.pos_encoding = positional_encoding() # set requires_grad = False

    def forward(self, x):
        #x += self.pos_encoding[:len(x)]
        logits = self.stack(x)
        return logits

x = torch.Tensor(3, d_model)
model = Transformer()
x = model(x)
print(x.shape)

