from __future__ import annotations

import numpy as np
import tqdm
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

#* ===========================================================================
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    seq_length: int = 1024
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    bias: bool = True
    drop_out: float = 0.2

#* ===========================================================================

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention that runs several different heads in parallel,
    each head learning a different relationships from the other heads
    """
    def __init__(self, config):
        """
        Initialize multi-head attention.

        Set up linear projections and validate configuration

        Args:
            embed_dim (int): Embedding dimension of the input and output tensors
            n_heads (int): Number of parallel heads to run in the attention mechanism
        """
        super().__init__()
        self.config = config
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads: int = config.num_heads
        self.embed_dim: int = config.embed_dim
        self.head_dim : int = config.embed_dim // config.num_heads
        
        #* query, key and values projections
        self.c_attn = nn.Linear(config.embed_dim , config.embed_dim * 3, bias = config.bias)
        
        #* output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        
        #* attention drop out and out put dropout
        self.attn_dropout = nn.Dropout(config.drop_out)
        self.res_dropout = nn.Dropout(config.drop_out)
        
        #* flash attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        q,k,v = self.c_attn(embedding).split(self.embed_dim, dim=2)
        
        B, T, C = q.shape       #* shape (batch, seq_len, embed_dim)
        #* split the heads      #* shape (batch, num_heads, seq_len, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        
        #* automatic self attention
        if self.flash:
            #* y.shape -> B, num_heads, T,T
            y = F.scaled_dot_product_attention(q, k, v, attn_mask= None, is_causal=True)
        else:
            #* manual implementation
            att = (q @ k.transpose(-1,-2)) / (self.embed_dim ** 0.5)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = torch.softmax(att, dim= -1, dtype = torch.float32)
            att = self.attn_dropout(att)
            y = att @ v
            
        #* merge all the heads back together
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        #* output projection
        y = self.res_dropout(self.c_proj(y))
    
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_dim, 4* config.embed_dim, bias =config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias= config.bias)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, X: torch.Tensor)-> torch.Tensor:
        X = X + self.attn(self.ln_1(X))
        X = X + self.mlp(self.ln_2(X))
        return X

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.seq_length, config.embed_dim),
            h = nn.ModuleList([Block(config)
                            for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _,t = idx.size()
        positions = torch.arange(0,t, dtype= torch.long)
        
        token_embed = self.transformer.wte(idx)
        pos_embed = self.transformer.wpe(positions)
        x = token_embed + pos_embed
        
        for layer in self.transformer.h:
            x = layer(x)
            
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_length: int,topk: int ) -> torch.Tensor:
        for _ in range(max_length):
            out = self(idx)
            out = out[:, -1, :]
            out = F.softmax(out, dim=-1)
            #* topk sampling
            top_probs, top_indices = torch.topk(out, k =topk, dim=-1)
            idx_next = torch.multinomial(top_probs.type(torch.float32), num_samples=1)
            xcol = torch.gather(top_indices, -1, idx_next)
            idx = torch.cat((idx, xcol), dim=1)
        return idx
        
    @classmethod
    def from_pretrained(cls, model_name: str ='gpt2') -> GPT2:
        print('⏳ Loading the pretrained weights from hugging face...')
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        #* get the model configuration
        config = {
            'gpt2': dict(embed_dim= 768, num_layers= 12, num_heads= 12),
            'gpt2-medium': dict(embed_dim= 1024, num_layers= 24, num_heads= 16),
            'gpt2-large': dict(embed_dim= 1280, num_layers= 36, num_heads= 20),
            'gpt2-xl': dict(embed_dim= 1600, num_layers= 48, num_heads= 25)
        }[model_name]
        
        config['vocab_size'] = 50257
        config['seq_length'] = 1024
        
        #* instantiate the model: both our model and hf model
        #* and get their state dictionaries
        config = GPT2Config(**config)
        model = GPT2(config)
        sd = model.state_dict()
        model_keys = sd.keys()
        model_keys = [key for key in model_keys if not key.endswith('.attn_bias')]
        
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        hf_sd = hf_model.state_dict()
        hf_keys = hf_sd.keys()
        hf_keys = [key for key in hf_keys if not key.endswith('.attn_masked_bias')]
        hf_keys = [key for key in hf_keys if not key.endswith('.attn_bias')]
        
        assert len(model_keys) == len(hf_keys), f'mismatched keys, expected {len(hf_keys)}, got {len(model_keys)}'
        #* transpose hugging face weights to match model weight shape
        transpose = ['c_attn.weight', 'c_proj.weight', 'c_fc.weight', 'c_proj.weight']
        
        for key in model_keys:
            if any(key.endswith(w) for w in transpose):
                if key not in hf_keys:
                    raise KeyError(
                        f"Key '{key}' not found in hugging face state dictionary"
                    )
                assert hf_sd[key].shape[::-1] == sd[key].shape, f'mismatched shapes, expected {sd[key].shape}, got {hf_sd[key].shape[::1]}'
                with torch.no_grad():
                    sd[key].copy_(hf_sd[key].T)
            else:
                assert hf_sd[key].shape == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(hf_sd[key])
                
        print('✅ Loaded pretrained weights from hugging face')
        print("🤗 Yaah, we didn't crash")
        return model

#* ===========================================================================

#? Data loading and training loop
with open('shakespeare.txt', 'r') as f:
    text = f.read()
    
n = int(0.9 * len(text))
train_data = text[:n]
val_data = text[n:]


eval_iters = 5
B, T = 4, 32
num_epochs = 50

class DataLoaderLite:
    def __init__(self, data, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(self.tokenizer.encode(data), dtype=torch.long) 

        len_data = len(self.tokens) 
        # number of batches
        self.n_batches = len_data // (batch_size * seq_length)

        # current position
        self.current_position = 0

    def next_batch(self): 
        B, T = self.batch_size, self.seq_length
        buf = self.tokens

        # Check if we need to reset for the next epoch
        if self.current_position + B * T + 1 > len(buf):
            self.current_position = 0

        # get a full batch
        x = buf[self.current_position : self.current_position + B*T].view(B,T)
        y = buf[self.current_position+1 : self.current_position + B*T + 1].view(B,T)

        # Advance current_position
        self.current_position += B*T

        return x,y

#* ================================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Create separate DataLoaderLite instances for train and validation
train_dataloader = DataLoaderLite(train_data, B, T)
print(f'Got {train_dataloader.n_batches} batches for train split')
val_dataloader = DataLoaderLite(val_data, B, T)
print(f'Got {val_dataloader.n_batches} batches for validation split')

torch.set_float32_matmul_precision('high')
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    #* Iterate over specific dataloaders for train and val splits
    for split, dataloader_obj in [('train', train_dataloader), ('val', val_dataloader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = dataloader_obj.next_batch()
            x,y = x.to(device), y.to(device)
            _, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#* =============================================================================

model = GPT2(GPT2Config())
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for iter in range(num_epochs):
    t0 = time.time()
    # zero gradients from previous iteration
    optimizer.zero_grad()

    # forward pass
    xb, yb = train_dataloader.next_batch() # Fixed: Use train_dataloader here
    xb, yb = xb.to(device), yb.to(device)

    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(xb, yb)

    # backward pass
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (B * T) / (dt / 1000)

    # evaluate the model
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}: val loss {losses['val']:.4f}: dt {dt:.4f}ms: token/sec {tokens_per_sec}")
