import torch
from torch import nn
import torch.nn.functional as F
import math
import random
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

d_model = 64
d_vocab = 10
ctx_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_example():
    length = random.randrange(1, 7)
    x = torch.zeros(ctx_size, dtype=torch.int32)
    x[0] = 1
    for i in range(1, length+1):
        x[i] = random.randrange(4, d_vocab)
    x[length+1] = 3
    for i in range(length+2, 2*length+2):
        x[i] = x[2*length+2-i]
    x[2*length+2] = 2
    y = torch.zeros(ctx_size, dtype=torch.long)
    for i in range(2*length+2):
        y[i] = x[i+1]
    return x, y

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
        self.pos_encoding = positional_encoding() # TODO: register as buffer
        self.sqrt = math.sqrt(d_model)

    def forward(self, index):
        x = torch.index_select(self.embedding, 0, index) * self.sqrt
        x += self.pos_encoding[:len(x)]
        x = self.decoders(x)
        logits = x @ self.embedding.T
        return logits


model = Transformer(num_layers=4, d_ff=4*d_model, h=4, d_k=16, d_v=16).to(device)
loss_fn = CrossEntropyLoss()
optim = Adam(model.parameters())

def train(num_examples=10000000):
    tot_loss, num_correct = 0, 0
    model.train()
    for i in range(num_examples):
        x, y = gen_example()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        mid_idx = (x == 3).nonzero(as_tuple=True)[0].item()
        tot_loss += loss.item()
        num_correct += torch.equal(pred.argmax(dim=1)[mid_idx:], y[mid_idx:])

        if i % 1000 == 999:
            print(f"Loss: {tot_loss/1000:.2f}, Accuracy: {num_correct/1000:.2f}")
            print(x)
            print(pred.argmax(dim=1))
            print(y)
            print()
            tot_loss = 0
            num_correct = 0

train()

