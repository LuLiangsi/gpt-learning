import torch
import torch.nn as nn
from torch.nn import functional as F
import time
torch.manual_seed(1337)



batch_size = 32 #parallel
block_size = 8
max_iter = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 200

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
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

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # generate 4 nums randomly
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')

# tell gpu we don't need to calculate the grad 
@torch.no_grad
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode to estimate the loss
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out

# here we just use one char to perdict the next char, the token do not talk to each other
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) 
        #(B, T, C) B batch T token_size in each batch C vocab_size

        if targets is None:
            loss = None
        else:
            #The reshaping is necessary because the cross-entropy loss function in PyTorch expects the input logits to have the shape (N, C), where N is the number of samples and C is the number of classes.
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #reshape the tensor into 2D
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] #take the last row of Token, become (B, C)
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1)
            # sample form the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim =1)# (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

start_time = time.time()

for iter in range(max_iter):

    if iter%eval_interval==0:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, value loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb) # output the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward() # get loss and optimize
    optimizer.step()

print(loss.item())


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

end_time = time.time()
print(f"用时：{(end_time-start_time):.2f}秒")