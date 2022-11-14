import torch
from torch import nn
import torch.nn.functional as F
import math
import random
from torch.nn import CrossEntropyLoss

d_model = 16
d_vocab = 10000
ctx_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_example():
    length = random.randrange(2, ctx_size)
    x = torch.zeros(length, d_model)
    start = random.randrange(0, d_model-length)
    for i in range(length):
        x[i][start+i] = 1
    return x[:-1], x[1:]

def positional_encoding():
    pos_enc = torch.Tensor(ctx_size, d_model)
    for pos in range(ctx_size):
        for i in range(d_model//2):
            pos_enc[pos][2*i] = math.sin(pos/10000**(2*i/d_model))
            pos_enc[pos][2*i+1] = math.cos(pos/10000**(2*i/d_model))
    return pos_enc.to(device)

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
        embedding = torch.empty(d_vocab, d_model)
        nn.init.kaiming_uniform_(embedding)
        self.embedding = nn.Parameter(embedding)
        layers = [DecoderBlock(p_drop=p_drop, d_ff=d_ff, h=h, d_k=d_k, d_v=d_v)
                  for _ in range(num_layers)]
        self.decoders = nn.Sequential(*layers)
        self.pos_encoding = positional_encoding()
        self.sqrt = math.sqrt(d_model)

    def forward(self, index):
        x = torch.index_select(self.embedding, 0, index) * self.sqrt
        x += self.pos_encoding[:len(x)]
        x = self.decoders(x)
        logits = x @ self.embedding.T
        return F.softmax(logits, dim=1)


model = Transformer().to(device)
loss_fn = CrossEntropyLoss()
x = torch.randint(d_vocab, (ctx_size,)).to(device)
x = model(x)
print(x)
print(x.shape)
print(x.sum())


"""
for i in range(1000):
    x, y = gen_example()
    x = model(x)
    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss.item())

"""
