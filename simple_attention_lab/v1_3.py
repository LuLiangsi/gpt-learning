import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import os


torch.manual_seed(1337)

path = os.path.dirname(os.path.abspath(__file__))

with open(path+'/threebody.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # generate 4 nums randomly
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    def __init__(self, num_embd, head_size, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(num_embd, head_size, bias=False)
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1)
        w = w * (q.shape[-1] ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.drop(w)
        out = w @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, num_embd, num_head, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(num_embd, num_embd // num_head, block_size, dropout) for _ in range(num_head)])
        self.linear = nn.Linear(num_embd // num_head * num_head, num_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.drop(self.linear(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, num_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embd, num_embd * 4),
            nn.ReLU(),
            nn.Linear(num_embd * 4, num_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, num_embd, num_head, block_size, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(num_embd)
        self.layernorm2 = nn.LayerNorm(num_embd)
        self.self_attn = MultiHead(num_embd, num_head, block_size, dropout)
        self.ff = FeedForward(num_embd, dropout)

    def forward(self, x):
        x = x + self.self_attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, num_embd, num_head, num_layer, block_size, vocab_size, dropout, device):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, num_embd)
        self.pos_embd = nn.Embedding(block_size, num_embd)
        self.blocks = nn.Sequential(*[Block(num_embd, num_head, block_size, dropout) for _ in range(num_layer)])
        self.layernorm = nn.LayerNorm(num_embd)
        self.linear = nn.Linear(num_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.device = device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T= x.shape

        x = self.token_embd(x) + self.pos_embd(torch.arange(T, device=self.device))
        x = self.blocks(x)
        x = self.layernorm(x)
        x = self.linear(x)
        
        return x
    
def generate(model, block_size, idx, max_new_tokens):
# size of idx is (B, T), T is the length of the sequence, B is the batch size.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits = model(idx_cond) # processed by forward
        # focus only on the last time step, because the last element the model output is the prediction of the next token of the whole sequence, as how we train the model
        logits = logits[:, -1, :] #take the last row of Token, become (B, C)
        # apply softmax to get probabilites
        probs = F.softmax(logits, dim=-1)
        # sample form the distribution
        idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
        # we can got the new token by decode the idx_next
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim =1)# (B, T+1)
    return idx

def generate_next(model, block_size, idx):
    idx_cond = idx[:, -block_size:]
    logits = model(idx_cond)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next
        
def train():
    batch_size = 16
    block_size = 64
    max_iters = 5
    eval_interval = 1
    out_interval = 3
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    n_embd = 64
    n_head = 6
    n_layer = 6
    dropout = 0.2
    
  
    model = GPT(n_embd, n_head, n_layer, block_size, vocab_size, dropout, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'num params: {num_params}')

    it_begin = time.time()

    for epoch in range(max_iters):
        

        model.train()
        x, y = get_batch('train', train_data, val_data, block_size, batch_size, device)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x, y = get_batch('val', train_data, val_data, block_size, batch_size, device)
                yhat = model(x)
                val_loss = F.cross_entropy(yhat.view(-1, vocab_size), y.view(-1))
                print(f'epoch {epoch} | val_loss {val_loss.item()} | time {time.time()-it_begin}')
                it_begin = time.time()
            
        if epoch % out_interval == 0:
            torch.save(model.state_dict(), path+f'/gpt_{epoch}.params')

