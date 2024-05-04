---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataclasses import dataclass
```

```python
torch.cuda.is_available()
```

```python
@dataclass
class ModelArgs:
    batch_size: int = 32
    d_model: int = 512
    hidden_dim: int = 1024 # hidden dim for feed forward layer
    num_blocks: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 16
    vocab_size: int = -1 # initialized later
    eps: float = 1e-6 # eps for RMSNorm
    max_batch_size: int = 32
    max_seq_len: int = 512
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq_base: int = 10000
    epochs: int = 5
```

```python
def compute_freqs(d_model:int, seq_len: int, device: torch.device | str, base: int):
    """
    d_model: embedding dim
    seq_len: sequence length
    device: cuda / cpu
    base: base for exponential of theta values
    """

    assert d_model % 2 == 0, "d_model has to be even"
    
    theta = 1. / (base ** (torch.arange(0, d_model, 2) / d_model)).to(device)
    m = torch.arange(seq_len).to(device)
    freqs = torch.outer(m, theta).float() # Since each m value corresponds to a single token, multiply every value of m by every value of theta, kind of like a nested for loop.
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # turn into complex form, z = r*cis(theta), in this case, r is 1
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: torch.device):
    """
    x: input sequence to add positional embedding, (batch, seq_len, emb_dim)
    freqs_complex: frequencies for rotary postitional embeddings
    device: cuda / cpu
    """
    # print(x.shape, freqs_complex.shape)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # (batch, seq_len, _, 2)
    # print(x_complex.shape)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # print(freqs_complex.shape)
    x_rotated = x_complex * freqs_complex # * is for element-wise multiplication
    # print(x_rotated.shape)
    x_rotated = torch.view_as_real(x_rotated)
    # print(x_rotated.shape)
    x_rotated = x_rotated.reshape(*x.shape)
    # print(x_rotated.shape)
    return x_rotated.type_as(x).to(device)

a = compute_freqs(32, 10, torch.device("cuda"), 10000)
print(a.shape)
b = torch.randn((3, 10, 5, 32)).to(torch.device("cuda"))
apply_rotary_embeddings(b, a, torch.device("cuda"))
```

```python
class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.eps = args.eps
        self.gamma = nn.Parameter(torch.ones(args.d_model))

    def rms(self, x: torch.Tensor):
        x = torch.pow(x, 2)
        x = torch.mean(x, dim=-1, keepdim=True)
        x = torch.sqrt(x + self.eps) # add eps to in case x = 0 (sqrt(0) is undefined in math)
        return x

    def forward(self, x: torch.Tensor):
        return x / self.rms(x) * self.gamma

args = ModelArgs(d_model=32)
r = RMSNorm(args)
a = torch.randn((5, 10, 32))
r(a).shape
```

```python
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.W = nn.Linear(args.d_model, args.hidden_dim)
        self.V = nn.Linear(args.d_model, args.hidden_dim)
        self.f = nn.Linear(args.hidden_dim, args.d_model)
    
    def forward(self, x: torch.Tensor):
        swiglu = F.silu(self.W(x)) * self.V(x)
        return self.f(swiglu)

l = FeedForward(args)
a = torch.randn((5, 10, 32))
l(a).shape
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_rep = args.num_q_heads // args.num_kv_heads
        
        self.q = nn.Linear(args.d_model, args.num_q_heads * args.d_model, bias=False)
        self.k = nn.Linear(args.d_model, args.num_kv_heads * args.d_model, bias=False)
        self.v = nn.Linear(args.d_model, args.num_kv_heads * args.d_model, bias=False)

        self.out = nn.Linear(args.num_q_heads * args.d_model, args.d_model, bias=False)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.num_kv_heads, args.d_model))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.num_kv_heads, args.d_model))

    
    def repeat_kv(self, x: torch.Tensor, n_rep: int):
        batch_size, seq_len, num_kv_heads, emb_dim = x.shape
        if n_rep == 1:
            return x
        return x[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, emb_dim).reshape(batch_size, seq_len, num_kv_heads * n_rep, emb_dim)

    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, use_kv_cache=False):
        batch_size, seq_len, _ = x.shape

        q = self.q(x) # (batch_size, seq_len, head_num, emb_dim)
        k = self.k(x)
        v = self.v(x)
        
        q = q.view(batch_size, seq_len, self.args.num_q_heads, self.args.d_model)
        k = k.view(batch_size, seq_len, self.args.num_kv_heads, self.args.d_model)
        v = v.view(batch_size, seq_len, self.args.num_kv_heads, self.args.d_model)

        q = apply_rotary_embeddings(q, freqs_complex, device=self.args.device)
        k = apply_rotary_embeddings(k, freqs_complex, device=self.args.device)
        
        keys, values = k, v
        if use_kv_cache:
            self.cache_k[:batch_size, start_pos: start_pos + seq_len] = k
            self.cache_v[:batch_size, start_pos: start_pos + seq_len] = v

            keys = self.cache_k[:batch_size, :start_pos + seq_len] # all cache including added key
            values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = self.repeat_kv(keys, self.num_rep)
        values = self.repeat_kv(values, self.num_rep)

        q = q.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 3, 1) # (batch_size, head_num, emb_dim, seq_len)
        
        scores = torch.matmul(q, keys) / self.args.d_model # (batch_size, head_num, seq_len, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        
        output = torch.matmul(scores, values) # (batch_size, head_num, seq_len, emb_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return self.out(output)
```

```python
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.norm1 = RMSNorm(args)
        self.norm2 = RMSNorm(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.norm1(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.norm2(h))
        return out
```

```python
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.embedding_layer = nn.Embedding(args.vocab_size, args.d_model)

        self.encoder_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.encoder_layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.freqs_complex = compute_freqs(args.d_model, args.max_seq_len, device=args.device, base=args.freq_base)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = self.embedding_layer(tokens)
        freqs_complex = self.freqs_complex[:, start_pos: start_pos + tokens.shape[1]]

        h = self.norm(h)

        for layer in self.encoder_layers:
            h = layer(h, start_pos, freqs_complex)

        output = self.output(h).float()
        return output
```

```python
import torch
import torch.nn as nn

class TransformerSparseEmbeddings(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # Use Sparse Embedding
        self.embedding_layer = nn.EmbeddingBag(args.vocab_size, args.d_model, sparse=True)

        self.encoder_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.encoder_layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args)
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.freqs_complex = compute_freqs(args.d_model, args.max_seq_len, device=args.device, base=args.freq_base)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # Modify the embedding lookup to work with sparse embeddings
        offsets = torch.arange(0, tokens.size(0) * tokens.size(1), tokens.size(1), device=tokens.device)
        h = self.embedding_layer(tokens.view(-1), offsets)

        h = h.view(tokens.size(0), tokens.size(1), -1)
        freqs_complex = self.freqs_complex[:, start_pos: start_pos + tokens.shape[1]]

        h = self.norm(h)

        for layer in self.encoder_layers:
            h = layer(h, start_pos, freqs_complex)

        output = self.output(h).float()
        return output
```

```python
import sentencepiece as spm
```
```python
import pandas as pd
df = pd.read_csv('./DeepLearning/Ja-En-LLaMA/en-ja.bicleaner05.txt', sep="\\t", header=None)
```

```python
df.head()[[3, 4]]
```

```python
df.info()
```

```python
df[3][9]
```

```python
# using trained tokenizers from http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/
english_tokenizer = spm.SentencePieceProcessor("./DeepLearning/Ja-En-LLaMA/enja_spm_models/spm.en.nopretok.model")
japanese_tokenizer = spm.SentencePieceProcessor("./DeepLearning/Ja-En-LLaMA/enja_spm_models/spm.ja.nopretok.model")
```

```python
df[3].isna().unique(), df[4].isna().unique()
```

```python
english_tokenizer.encode("Go to the original video hierarchy of the conversion source, copy and paste the following is fine. ffmpeg -i sample.mp4 -strict -2 video.webm summary I’ve been using the upload and embed method to Youtube to set up videos on the web.", out_type=str)
```

```python
japanese_tokenizer.encode("年金 日本に住んでいる20歳~60歳の全ての人は、公的年金制度に加入しなければなりません。", out_type=str)
```

```python
english_tokenizer.vocab_size(), japanese_tokenizer.vocab_size()
```

```python
from collections import Counter
from torchtext.vocab import vocab


def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer.encode(sentence, out_type=str))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>']) # specials: include special tokens in the mapping
```

```python
print(len(df), len(df) // 10)
```

```python
df = df.sample(frac=1)
english = df[3].tolist()[:len(df) // 500]
japanese = df[4].tolist()[:len(df) // 500]
assert len(english) == len(japanese)
l = len(english)
train_en = english[:int(0.7 * l)]
val_en = english[int(0.7 * l): int(0.85 * l)]
test_en = english[int(0.85 * l): l]
train_ja = japanese[:int(0.7 * l)]
val_ja = japanese[int(0.7 * l): int(0.85 * l)]
test_ja = japanese[int(0.85 * l): l]
```

```python
print(len(train_en), len(df))
```

```python
vocab_ja = build_vocab(japanese, japanese_tokenizer)
vocab_en = build_vocab(english, english_tokenizer)
```

```python
def data_process(ja, en):
    data = []
    for (raw_ja, raw_en) in zip(ja, en):
        ja_tensor = torch.tensor([vocab_ja[token] for token in japanese_tokenizer.encode(raw_ja.strip("\n"), out_type=str)], dtype=torch.long)
        en_tensor = torch.tensor([vocab_en[token] for token in english_tokenizer.encode(raw_en.rstrip("\n"), out_type=str)], dtype=torch.long)
        data.append((ja_tensor, en_tensor))
    return data
```

```python
train = data_process(train_ja, train_en)
```

```python
a = ModelArgs()
print(vars(a))
```

```python
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = vocab_ja['<pad>']
BOS_IDX = vocab_ja['<bos>']
EOS_IDX = vocab_ja['<eos>']

def generate_batch(data_batch):
    ja_batch, en_batch = [], []
    for (ja_item, en_item) in data_batch:
        ja_batch.append(torch.cat([torch.tensor([BOS_IDX]), ja_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    ja_batch = pad_sequence(ja_batch, padding_value=PAD_IDX) # pad sequences into equal length
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return ja_batch, en_batch
```

```python
from torch.utils.data import DataLoader

args = ModelArgs()

train_iter = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=generate_batch) # type: ignore
```

```python
len(vocab_ja)
```


```python
args.vocab_size = len(vocab_ja)
# args = ModelArgs(batch_size=32, d_model=64, hidden_dim=512, num_blocks=8, num_q_heads=32, num_kv_heads=16, vocab_size=100)
print(vars(args))
```

```python
args = ModelArgs(batch_size=32, d_model=64, hidden_dim=512, num_blocks=8, num_q_heads=32, num_kv_heads=16, vocab_size=len(vocab_ja))
transformer = TransformerSparseEmbeddings(args)
transformer = transformer.to(args.device)
```

```python
for (i, j) in train_iter:
    print(i.shape, j.shape)
    break
```

```python
len(train_iter)
```

```python
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)

def train_epoch(model, train_iter, optimizer, device): # https://chatgpt.com/c/2be13c26-50d8-4583-97b6-397c1fe2d028
    model.train()
    losses = 0
    for i, (x, y) in enumerate(train_iter):
        x = x.to(device)
        y = y.to(device)
        y = y[:-1, :]
        logits = model(x, 0)

        y_out = y[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)
```

```python

def evaluate(model, val_iter, device):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(val_iter)):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        logits = model(src, 0)
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)
```

```python
import time
for epoch in range(1, args.epochs+1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer, args.device)
    end_time = time.time()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s"))
```

```python

```

