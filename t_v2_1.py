from v2_1 import *
import sys


batch_size = 64
block_size = 256
max_iters = 50000
eval_interval = 500
out_interval = 2000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps'
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.1

model = GPT(n_embd, n_head, n_layer, block_size, vocab_size, dropout, device).to(device)
model.load_state_dict(torch.load('gpt_12000.params', map_location=device))

model.eval()
while True:
    inp = input('You >>> ')
    if inp == 'e':
        break
    context = torch.tensor([encode(inp)]).to(device)
    next_text = context

    output = ''
    sys.stdout.write('Bot >>> ')

    newline = 0

    while newline < 3:
        output = decode(next_text[0].tolist())
        if output[-1] == '\n':
            newline += 1
        
        sys.stdout.write(output)
        sys.stdout.flush()

        next_text = generate_next(model, block_size, context)
        context = torch.cat([context, next_text], dim=1)

    