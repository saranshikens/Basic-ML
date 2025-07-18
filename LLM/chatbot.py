import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
args = parser.parse_args()



block_size = 32
batch_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2



chars = ""
with open('OpenWebText/vocab.txt', 'r', encoding='utf-8') as f: 
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)



string_to_int = {char:i for i,char in enumerate(chars)}
int_to_string = {i:char for i,char in enumerate(chars)}
encode = lambda string: [string_to_int[char] for char in string]
decode = lambda encoded_string: ''.join([int_to_string[i] for i in encoded_string])



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = X.shape
        k = self.key(X) # (B,T,hs)
        q = self.query(X) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q@k.transpose(-2,-1)*k.shape[-1]**(-0.5) # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        V = self.value(X) # (B,T,hs)
        out = wei @ V # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # difference between Module List and Sequential :-
        # Module List -> attention for multiple heads are calculated parallely
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1) # concatenate along last dimension i.e. channel. 
        # (B,T,C) -> (B,T, [h1,h1,h1,....,h2,h2,h2,.....,h3,h3,h3,......]), where h1's are features of 1st head
        out = self.dropout(self.proj(out))
        return out
    


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.net(X)
    


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, X):
        y = self.sa(X)
        X = self.ln1(X+y)
        y = self.ffwd(X)
        X = self.ln2(X+y)
        return X
    


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, index, targets=None):
        B,T = index.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # become (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # sample from the distribution
            index = torch.cat((index, index_next), dim=-1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
print("loading model parameters")

with open('model-01.pkl', 'rb') as f:
    model = pickle.load()
print("loaded")



while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')


