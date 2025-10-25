/**
 * Transformer Architecture Content
 */

export const transformerArchitectureSection = {
  id: 'transformer-architecture',
  title: 'Transformer Architecture',
  content: `
# Transformer Architecture

## Introduction

The **Transformer** architecture, introduced in "Attention Is All You Need" (2017), revolutionized deep learning by completely **eliminating recurrence and convolutions**, relying entirely on attention mechanisms.

**Why Transformers?**

- **Parallelization**: RNNs process sequentially (h_t depends on h_{t-1}), limiting GPU utilization. Transformers process all positions **simultaneously**
- **Long-range dependencies**: Self-attention directly connects any two positions, regardless of distance
- **State-of-the-art**: Transformers power GPT, BERT, T5, and virtually all modern NLP models

**Key Innovation**: **Self-attention** - Each position attends to ALL other positions in the same sequence to compute its representation.

---

## Core Architecture Components

### 1. Self-Attention Mechanism

**Unlike Seq2Seq attention** (decoder → encoder), **self-attention** computes attention **within the same sequence**.

**Process**:

1. **Input embeddings**: Convert tokens to vectors **x_1, ..., x_n** (each d_model dimensional)
2. **Linear projections**: Create Q, K, V for each position

\`\`\`
Q_i = x_i W^Q  (query: "what am I looking for?")
K_i = x_i W^K  (key: "what do I contain?")
V_i = x_i W^V  (value: "what information do I provide?")
\`\`\`

3. **Attention scores**: Measure similarity between Q_i and all keys

\`\`\`
score(Q_i, K_j) = Q_i · K_j / √d_k
\`\`\`

Scaling by √d_k prevents dot products from growing too large.

4. **Attention weights**: Softmax over scores

\`\`\`
α_{i,j} = softmax_j (score(Q_i, K_j))
\`\`\`

5. **Output**: Weighted sum of values

\`\`\`
output_i = Σ_j α_{i,j} V_j
\`\`\`

**Example**: "The cat sat on the mat"

For position "sat":
- High attention to "cat" (subject)
- High attention to "mat" (object)
- Moderate attention to "on" (preposition)
- Low attention to "the" (less semantic content)

Self-attention allows "sat" to see the entire sentence context!

---

### 2. Multi-Head Attention

**Limitation of single attention**: One attention distribution must capture all relationships (syntactic, semantic, positional).

**Solution**: **Multiple attention heads** learn different aspects.

**Architecture**:

\`\`\`
MultiHead(Q, K, V) = Concat (head_1, ..., head_h) W^O

where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
\`\`\`

**Parameters**:
- h = 8 heads (original paper)
- d_model = 512 (full dimension)
- d_k = d_v = 64 (per-head dimension = 512/8)

**What different heads learn**:
- Head 1: Syntactic dependencies (subject-verb, verb-object)
- Head 2: Positional relationships (adjacent words)
- Head 3: Semantic similarities (synonyms, related concepts)
- Head 4: Long-range dependencies (pronoun resolution)
- Heads 5-8: Task-specific patterns

**Analogy**: Like having multiple perspectives on the same data.

---

### 3. Positional Encoding

**Problem**: Self-attention is **permutation invariant** - order doesn't matter!

"cat the on mat sat" would produce the same representation as "the cat sat on the mat".

**Solution**: Add **positional encodings** to input embeddings to inject position information.

**Sinusoidal encoding** (original Transformer):

\`\`\`
PE(pos, 2i) = sin (pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos (pos / 10000^(2i/d_model))
\`\`\`

**Properties**:
- Deterministic (not learned)
- Unique for each position
- Can extrapolate to longer sequences
- Encodes relative positions: PE(pos+k) is a linear function of PE(pos)

**Alternative**: **Learned positional embeddings** (used in BERT, GPT)
- Pros: Potentially better adapted to task
- Cons: Can't handle sequences longer than training saw

---

### 4. Feed-Forward Networks

After attention, each position passes through an **identical feed-forward network** (applied independently):

\`\`\`
FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
\`\`\`

**Architecture**:
- Input dimension: d_model = 512
- Hidden dimension: d_ff = 2048 (4× expansion)
- Output dimension: d_model = 512

**Purpose**:
- Add **non-linearity** (attention is linear operations)
- Increase **representational capacity**
- Process each position's aggregated information independently

Think of it as: "After seeing the context (attention), what do I do with it?"

---

### 5. Layer Normalization & Residual Connections

Each sub-layer (attention, FFN) uses:

\`\`\`
LayerNorm (x + Sublayer (x))
\`\`\`

**Residual connection** (x +): 
- Helps gradient flow in deep networks
- Allows lower layers to pass information directly to higher layers

**Layer Normalization**: 
- Normalizes activations per example across features
- Stabilizes training (vs. batch normalization which normalizes across batch)

---

## Complete Transformer Architecture

### Encoder

**Single encoder layer**:
1. Multi-head self-attention
2. Add & normalize
3. Feed-forward network
4. Add & normalize

**Full encoder**: Stack of N = 6 identical layers

**Input**: Token embeddings + positional encodings
**Output**: Contextualized representations for each token

### Decoder

**Single decoder layer**:
1. **Masked** multi-head self-attention (can't see future)
2. Add & normalize
3. Multi-head **cross-attention** (queries from decoder, keys/values from encoder)
4. Add & normalize
5. Feed-forward network
6. Add & normalize

**Full decoder**: Stack of N = 6 identical layers

### Complete Model

**Encoder → Decoder**:
- Encoder processes source sequence
- Decoder generates target sequence autoregressively
- Cross-attention connects encoder and decoder

**Output layer**: Linear + softmax to predict next token probability distribution

---

## Mathematical Formulation

### Scaled Dot-Product Attention

\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) V
\`\`\`

**Matrix dimensions** (batch_size × seq_len × d_k):
- Q: queries (n × d_k)
- K: keys (m × d_k)
- V: values (m × d_v)
- QK^T: (n × m) attention scores
- After softmax: (n × m) attention weights
- Output: (n × d_v)

### Multi-Head Attention

\`\`\`
MultiHead(Q, K, V) = Concat (head_1, ..., head_h) W^O

head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
\`\`\`

**Parameter count**:
- Per head: 3 × d_model × d_k = 3 × 512 × 64 = 98,304
- All heads: 8 × 98,304 = 786,432
- Output projection W^O: 512 × 512 = 262,144
- **Total**: ~1M parameters per attention layer

---

## Implementation in PyTorch

### Scaled Dot-Product Attention

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Queries (batch_size, num_heads, seq_len_q, d_k)
        K: Keys (batch_size, num_heads, seq_len_k, d_k)
        V: Values (batch_size, num_heads, seq_len_k, d_v)
        mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
    
    Returns:
        output: (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores: QK^T / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt (d_k)
    # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    
    # Apply mask (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill (mask == 0, -1e9)
    
    # Compute attention weights
    attention_weights = F.softmax (scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul (attention_weights, V)
    
    return output, attention_weights


# Example usage
batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64

Q = torch.randn (batch_size, num_heads, seq_len, d_k)
K = torch.randn (batch_size, num_heads, seq_len, d_k)
V = torch.randn (batch_size, num_heads, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (2, 8, 10, 64)
print(f"Attention weights shape: {attn_weights.shape}")  # (2, 8, 10, 10)
print(f"Attention weights sum: {attn_weights[0, 0, 0].sum()}")  # Should be 1.0
\`\`\`

### Multi-Head Attention

\`\`\`python
class MultiHeadAttention (nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Model dimension (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear (d_model, d_model)
        self.W_k = nn.Linear (d_model, d_model)
        self.W_v = nn.Linear (d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear (d_model, d_model)
        
    def split_heads (self, x):
        """Split last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.view (batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads (self, x):
        """Inverse of split_heads"""
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.contiguous().view (batch_size, seq_len, self.d_model)
    
    def forward (self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: Optional (batch_size, 1, 1, seq_len_k)
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len_q, d_model)
        K = self.W_k(K)  # (batch, seq_len_k, d_model)
        V = self.W_v(V)  # (batch, seq_len_k, d_model)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_len_k, d_k)
        
        # Scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, seq_len_q, d_k)
        
        # Combine heads
        output = self.combine_heads (attn_output)  # (batch, seq_len_q, d_model)
        
        # Final linear projection
        output = self.W_o (output)
        
        return output, attn_weights


# Example usage
d_model, num_heads, seq_len = 512, 8, 10
batch_size = 2

mha = MultiHeadAttention (d_model, num_heads)

# Self-attention (Q=K=V)
x = torch.randn (batch_size, seq_len, d_model)
output, attn_weights = mha (x, x, x)

print(f"Input shape: {x.shape}")  # (2, 10, 512)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Attention weights shape: {attn_weights.shape}")  # (2, 8, 10, 10)
\`\`\`

### Positional Encoding

\`\`\`python
class PositionalEncoding (nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp (torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin (position * div_term)
        pe[:, 1::2] = torch.cos (position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward (self, x):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
        
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# Visualize positional encodings
import matplotlib.pyplot as plt

d_model = 128
max_len = 100
pe_layer = PositionalEncoding (d_model, max_len)

# Get positional encodings (for a dummy input)
x = torch.zeros(1, max_len, d_model)
pe = pe_layer.pe[0].numpy()  # (max_len, d_model)

plt.figure (figsize=(12, 6))
plt.imshow (pe.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding Visualization')
plt.colorbar()
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
print("Saved positional encoding visualization")

# Plot specific dimensions
plt.figure (figsize=(12, 6))
for i in [0, 1, 4, 8, 16, 32]:
    plt.plot (pe[:, i], label=f'Dim {i}')
plt.xlabel('Position')
plt.ylabel('Value')
plt.title('Positional Encoding - Selected Dimensions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('positional_encoding_dimensions.png', dpi=150, bbox_inches='tight')
print("Saved dimension-specific visualization")
\`\`\`

### Feed-Forward Network

\`\`\`python
class FeedForward (nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension (e.g., 512)
            d_ff: Hidden dimension (e.g., 2048)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear (d_model, d_ff)
        self.dropout = nn.Dropout (dropout)
        self.linear2 = nn.Linear (d_ff, d_model)
    
    def forward (self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Expand: d_model -> d_ff
        x = self.linear1(x)
        x = F.relu (x)
        x = self.dropout (x)
        
        # Contract: d_ff -> d_model
        x = self.linear2(x)
        
        return x


# Example usage
d_model, d_ff = 512, 2048
batch_size, seq_len = 2, 10

ff = FeedForward (d_model, d_ff)
x = torch.randn (batch_size, seq_len, d_model)
output = ff (x)

print(f"Input shape: {x.shape}")  # (2, 10, 512)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Parameters: {sum (p.numel() for p in ff.parameters()):,}")
# ~2.1M parameters per FFN layer
\`\`\`

### Encoder Layer

\`\`\`python
class EncoderLayer (nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single Transformer encoder layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention (d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward (d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm (d_model)
        self.norm2 = nn.LayerNorm (d_model)
        
        # Dropout
        self.dropout = nn.Dropout (dropout)
    
    def forward (self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Multi-head attention + residual + norm
        attn_output, _ = self.self_attn (x, x, x, mask)
        x = self.norm1(x + self.dropout (attn_output))
        
        # Feed-forward + residual + norm
        ff_output = self.feed_forward (x)
        x = self.norm2(x + self.dropout (ff_output))
        
        return x


class TransformerEncoder (nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Stack of encoder layers
        
        Args:
            num_layers: Number of encoder layers (e.g., 6)
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer (d_model, num_heads, d_ff, dropout)
            for _ in range (num_layers)
        ])
    
    def forward (self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer (x, mask)
        
        return x


# Example usage
num_layers, d_model, num_heads, d_ff = 6, 512, 8, 2048
batch_size, seq_len = 2, 10

encoder = TransformerEncoder (num_layers, d_model, num_heads, d_ff)
x = torch.randn (batch_size, seq_len, d_model)
output = encoder (x)

print(f"Input shape: {x.shape}")  # (2, 10, 512)
print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Total parameters: {sum (p.numel() for p in encoder.parameters()):,}")
# ~19M parameters for 6-layer encoder
\`\`\`

### Complete Transformer for Translation

\`\`\`python
class Transformer (nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        """
        Complete Transformer model for sequence-to-sequence tasks
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding (src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding (tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding (d_model, max_len)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder (num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder (num_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_proj = nn.Linear (d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout (dropout)
    
    def encode (self, src, src_mask=None):
        """Encode source sequence"""
        # Embedding + positional encoding
        src = self.src_embedding (src) * math.sqrt (self.d_model)
        src = self.pos_encoding (src)
        src = self.dropout (src)
        
        # Encode
        memory = self.encoder (src, src_mask)
        return memory
    
    def decode (self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decode target sequence"""
        # Embedding + positional encoding
        tgt = self.tgt_embedding (tgt) * math.sqrt (self.d_model)
        tgt = self.pos_encoding (tgt)
        tgt = self.dropout (tgt)
        
        # Decode
        output = self.decoder (tgt, memory, tgt_mask, memory_mask)
        
        # Project to vocabulary
        logits = self.output_proj (output)
        return logits
    
    def forward (self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            memory_mask: Encoder-decoder mask
        
        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Encode source
        memory = self.encode (src, src_mask)
        
        # Decode target
        logits = self.decode (tgt, memory, tgt_mask, memory_mask)
        
        return logits


# Helper function to create causal mask
def create_causal_mask (size):
    """Create mask that prevents attending to future positions"""
    mask = torch.triu (torch.ones (size, size), diagonal=1).bool()
    return ~mask  # Invert: True where we CAN attend


# Example usage
src_vocab_size, tgt_vocab_size = 10000, 8000
d_model, num_heads, num_layers, d_ff = 512, 8, 6, 2048

model = Transformer (src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                    num_layers, d_ff)

# Dummy data
batch_size = 2
src = torch.randint(0, src_vocab_size, (batch_size, 20))  # Source: length 20
tgt = torch.randint(0, tgt_vocab_size, (batch_size, 15))  # Target: length 15

# Create causal mask for decoder
tgt_mask = create_causal_mask(15).unsqueeze(0).unsqueeze(0)
# Shape: (1, 1, 15, 15) - broadcasts over batch and heads

# Forward pass
logits = model (src, tgt, tgt_mask=tgt_mask)

print(f"Source shape: {src.shape}")  # (2, 20)
print(f"Target shape: {tgt.shape}")  # (2, 15)
print(f"Output logits shape: {logits.shape}")  # (2, 15, 8000)
print(f"Total parameters: {sum (p.numel() for p in model.parameters()):,}")
# ~65M parameters for base Transformer
\`\`\`

---

## Training Transformers

### Loss Function

Cross-entropy loss over predicted token distributions:

\`\`\`python
criterion = nn.CrossEntropyLoss (ignore_index=PAD_IDX)

# Forward pass
logits = model (src, tgt[:, :-1], tgt_mask=tgt_mask)  # Teacher forcing
# Predict tokens 1...n from tokens 0...n-1

# Reshape for loss computation
logits = logits.reshape(-1, tgt_vocab_size)  # (batch * seq_len, vocab)
targets = tgt[:, 1:].reshape(-1)  # (batch * seq_len)

# Compute loss
loss = criterion (logits, targets)
\`\`\`

### Learning Rate Scheduling

Original Transformer used **warm-up scheduler**:

\`\`\`
lr = d_model^{-0.5} × min (step^{-0.5}, step × warmup_steps^{-1.5})
\`\`\`

- Learning rate **increases** during warm-up (first 4000 steps)
- Then **decreases** proportionally to inverse square root of step number

\`\`\`python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step (self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr (self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )


# Example usage
optimizer = torch.optim.Adam (model.parameters(), betas=(0.9, 0.98), eps=1e-9)
scheduler = TransformerLRScheduler (optimizer, d_model=512, warmup_steps=4000)

# Training loop
for epoch in range (num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        logits = model (batch.src, batch.tgt[:, :-1])
        loss = criterion (logits.reshape(-1, tgt_vocab_size), 
                        batch.tgt[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        scheduler.step()
\`\`\`

---

## Real-World Applications

### 1. Machine Translation

**Task**: English → German translation

**Architecture**:
- Encoder: 6 layers, processes English sentence
- Decoder: 6 layers, generates German translation
- Cross-attention: Decoder attends to English encoder states

**Results** (WMT'14 En-De):
- Transformer: 28.4 BLEU
- Previous best (ensemble): 26.3 BLEU
- **2.1 BLEU improvement** + faster training!

### 2. Text Summarization

**Task**: Long document → Short summary

**Why Transformers work well**:
- Long-range dependencies: Connect summary to distant document parts
- Parallelization: Fast processing of long documents
- Attention visualization: Shows which parts influenced summary

**Example** (CNN/DailyMail dataset):
- Input: 800-word news article
- Output: 3-sentence summary
- ROUGE-L score: 41.2 (vs. 39.5 for RNN)

### 3. Question Answering

**Task**: Given passage + question → Answer span

**Architecture**:
- Encode passage and question jointly
- Self-attention learns relationships
- Output: Start and end positions of answer

**SQuAD 2.0 results**:
- BERT (Transformer-based): 89.8 F1
- Human performance: 89.5 F1

---

## Advantages Over RNNs

1. **Parallelization**: All positions processed simultaneously (vs. sequential in RNNs)
   - Training time: 100K steps in 12 hours (8 GPUs)
   - RNN equivalent: 3-5 days

2. **Long-range dependencies**: Direct connections between any positions
   - Path length: O(1) (vs. O(n) in RNN)
   - No vanishing gradient over distance

3. **Interpretability**: Attention weights show which positions model focuses on
   - Can visualize what model "thinks"
   - Helps debugging and analysis

4. **Scalability**: Performance improves with model size
   - GPT-3: 175B parameters
   - Not practical with RNNs

---

## Limitations & Solutions

### 1. Computational Complexity

**Problem**: Self-attention is O(n²) in sequence length

**Solutions**:
- **Sparse attention**: Attend to subset of positions (Sparse Transformer)
- **Local attention**: Attend to nearby positions only (Longformer)
- **Linear attention**: Approximate attention in O(n) (Performer, Linear Transformer)

### 2. Limited Context Window

**Problem**: Fixed maximum length (512 tokens for BERT)

**Solutions**:
- **Sliding window**: Process long documents in chunks
- **Hierarchical models**: Sentence-level → document-level
- **Recurrence**: Transformer-XL adds recurrence for unbounded context

### 3. Data Efficiency

**Problem**: Needs large datasets to train from scratch

**Solutions**:
- **Pre-training**: Train on massive unlabeled data, then fine-tune
- **Transfer learning**: Use pre-trained models (BERT, GPT, T5)
- **Few-shot learning**: GPT-3 style prompting

---

## Discussion Questions

1. **Why is positional encoding necessary in Transformers but not in RNNs?**
   - Consider how each architecture processes sequences

2. **Multi-head attention uses 8 heads with d_k = 64 each, rather than 1 head with d_k = 512. What are the advantages of multiple smaller heads?**
   - Think about what different heads might learn

3. **The Transformer encoder is fully parallelizable, but the decoder still generates autoregressively (one token at a time). Why can't we parallelize decoding?**
   - Consider the causal dependency structure

4. **Transformers achieve state-of-the-art results on many NLP tasks. Why haven't they completely replaced RNNs for all sequence tasks?**
   - Consider computational costs, data requirements, and specific task characteristics

5. **How does the attention mechanism in Transformers differ from human attention when reading text?**
   - Compare the "soft" weighted attention to human "hard" selective attention

---

## Key Takeaways

- **Self-attention** allows each position to attend to all positions in the same sequence, capturing relationships directly
- **Multi-head attention** learns different aspects of relationships (syntactic, semantic, positional) in parallel
- **Positional encoding** injects position information since attention is permutation-invariant
- **Feed-forward networks** add non-linearity and process aggregated context independently per position
- **Layer normalization & residual connections** stabilize training in deep networks (6+ layers)
- **Encoder-decoder architecture**: Encoder processes source, decoder generates target, cross-attention connects them
- **Parallelization** enables training on much larger datasets in reasonable time (vs. RNNs)
- **Scalability**: Transformers improve with size (GPT-3: 175B parameters)
- **Limitations**: O(n²) complexity, fixed context window, data-hungry - active research area
- **Impact**: Foundation of modern NLP (BERT, GPT, T5) and expanding to vision (ViT), speech, multimodal

---

## Practical Tips

1. **Start with pre-trained models**: Don't train from scratch unless you have massive data and compute

2. **Use appropriate positional encoding**: Sinusoidal for variable lengths, learned for fixed

3. **Implement masking carefully**: Padding mask (ignore), causal mask (no future), combined masks

4. **Monitor attention weights**: Visualize to understand what model learned and debug issues

5. **Tune carefully**: Learning rate scheduling crucial, warm-up prevents early instability

6. **Gradient clipping**: Prevents exploding gradients, especially early in training

7. **Memory management**: Attention is O(n²), batch size × sequence length matters significantly

8. **Consider efficient variants**: For long sequences (>512), use Longformer, BigBird, or Performer

---

## Further Reading

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Original Transformer paper (Vaswani et al., 2017)
- ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/) - Excellent visual guide
- ["The Annotated Transformer"](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained models and tools

---

*Next Section: Transfer Learning - Learn how to leverage pre-trained Transformers (BERT, GPT) for your specific tasks with minimal data!*
`,
};
