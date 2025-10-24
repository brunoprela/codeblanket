export const transformerArchitectureDeepDive = {
  title: 'Transformer Architecture Deep Dive',
  id: 'transformer-architecture-deep-dive',
  content: `
# Transformer Architecture Deep Dive

## Introduction

The Transformer architecture, introduced in "Attention is All You Need" (2017), revolutionized NLP and became the foundation for all modern LLMs. Unlike RNNs that process sequentially, transformers process entire sequences in parallel using self-attention mechanisms.

This section provides a comprehensive, from-scratch understanding of every component: multi-head attention, positional encodings, feed-forward networks, layer normalization, and how they combine to create powerful language models.

### Why Transformers Won

**Parallelization**: Process entire sequence at once (vs sequential RNNs)
**Long-range dependencies**: Attention connects any two positions directly
**Scalability**: Architecture scales to billions of parameters
**Interpretability**: Attention weights show what model focuses on
**Transfer learning**: Pre-trained transformers transfer across tasks

---

## Self-Attention Mechanism

### Understanding Attention

\`\`\`python
"""
Self-attention from scratch
"""

import numpy as np
import torch
import torch.nn as nn

# Intuition: Attention lets each word "look at" other words
# Example: "The cat sat on the mat"
# When processing "sat", attention might focus on "cat" (who sat?)

def simple_attention(query, key, value):
    """
    Simplified attention mechanism
    
    Args:
        query: What we're looking for [seq_len, d_model]
        key: What we're looking in [seq_len, d_model]
        value: What we want to get [seq_len, d_model]
    
    Returns:
        attention output [seq_len, d_model]
    """
    # 1. Compute attention scores: how relevant is each key to each query?
    scores = np.matmul(query, key.T)  # [seq_len, seq_len]
    # scores[i, j] = relevance of position j to position i
    
    # 2. Scale scores (prevent large values)
    d_k = query.shape[-1]
    scores = scores / np.sqrt(d_k)
    
    # 3. Softmax: convert to probabilities
    attention_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    # attention_weights[i] = probability distribution over all positions
    
    # 4. Weighted sum of values
    output = np.matmul(attention_weights, value)
    # output[i] = weighted average of all value vectors
    
    return output, attention_weights

# Example: Self-attention in practice
sentence = "The cat sat on the mat"
tokens = sentence.split()
d_model = 512

# Create dummy embeddings
embeddings = np.random.randn(len(tokens), d_model)

# In self-attention: query = key = value = embeddings
output, attention_weights = simple_attention(embeddings, embeddings, embeddings)

print("Attention weights shape:", attention_weights.shape)  # [6, 6]
print("\\nAttention weights (token -> token):")
for i, token in enumerate(tokens):
    print(f"{token}: {attention_weights[i]}")
    # Shows which tokens this token attends to

# Interpretation example:
# When processing "sat", attention might be:
# [0.05, 0.60, 0.20, 0.05, 0.05, 0.05]
#  The   cat   sat   on   the   mat
# → Model focuses most on "cat" (the subject who sat)
\`\`\`

### Scaled Dot-Product Attention

\`\`\`python
"""
Complete scaled dot-product attention implementation
"""

import torch
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, n_heads, seq_len, d_k]
            key: [batch, n_heads, seq_len, d_k]
            value: [batch, n_heads, seq_len, d_v]
            mask: [batch, 1, seq_len, seq_len] (optional)
        
        Returns:
            output: [batch, n_heads, seq_len, d_v]
            attention_weights: [batch, n_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # 1. Compute attention scores
        # QK^T: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. Scale by sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # Why scale?
        # For large d_k, dot products grow large
        # Large values → softmax saturates → small gradients
        # Scaling keeps variance stable
        
        # 3. Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            # Masked positions get -inf → softmax = 0
        
        # 4. Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # 6. Weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Usage example
batch_size = 32
n_heads = 8
seq_len = 50
d_k = 64

attention = ScaledDotProductAttention()

query = torch.randn(batch_size, n_heads, seq_len, d_k)
key = torch.randn(batch_size, n_heads, seq_len, d_k)
value = torch.randn(batch_size, n_heads, seq_len, d_k)

output, weights = attention(query, key, value)

print(f"Output shape: {output.shape}")  # [32, 8, 50, 64]
print(f"Attention weights: {weights.shape}")  # [32, 8, 50, 50]
\`\`\`

---

## Multi-Head Attention

### Why Multiple Heads?

\`\`\`python
"""
Multi-head attention: Multiple attention mechanisms in parallel
"""

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows model to attend to different aspects
    
    Example: When processing "The cat sat on the mat"
    - Head 1: Focus on subject-verb relationships (cat -> sat)
    - Head 2: Focus on prepositions (sat -> on, on -> mat)
    - Head 3: Focus on articles (the -> cat, the -> mat)
    - etc.
    
    Different heads learn different patterns!
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        Split last dimension into (n_heads, d_k)
        
        Input: [batch, seq_len, d_model]
        Output: [batch, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back to d_model
        
        Input: [batch, n_heads, seq_len, d_k]
        Output: [batch, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # [batch, n_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: [batch, n_heads, seq_len, d_k]
        
        # 4. Combine heads
        output = self.combine_heads(attn_output)  # [batch, seq_len, d_model]
        
        # 5. Final linear projection
        output = self.W_o(output)
        output = self.dropout(output)
        
        return output, attn_weights

# Example usage
d_model = 512
n_heads = 8
batch_size = 32
seq_len = 50

mha = MultiHeadAttention(d_model, n_heads)

x = torch.randn(batch_size, seq_len, d_model)
output, weights = mha(x, x, x)

print(f"Input: {x.shape}")  # [32, 50, 512]
print(f"Output: {output.shape}")  # [32, 50, 512]
print(f"Attention per head: {weights.shape}")  # [32, 8, 50, 50]

# Visualize attention patterns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for head in range(8):
    ax = axes[head // 4, head % 4]
    ax.imshow(weights[0, head].detach().numpy(), cmap='viridis')
    ax.set_title(f'Head {head + 1}')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
plt.tight_layout()
plt.show()

# You'll see different patterns in different heads!
\`\`\`

---

## Positional Encodings

### Why Positional Information Matters

\`\`\`python
"""
Positional encodings: Teaching transformers about token order
"""

# Problem: Self-attention is permutation invariant
# "cat sat mat" and "mat sat cat" produce same output!
# Need to inject position information

class PositionalEncoding(nn.Module):
    """
    Add position information using sine/cosine functions
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos = position, i = dimension
    """
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + positional encoding: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Visualize positional encodings
import matplotlib.pyplot as plt

d_model = 512
max_seq_len = 100

pos_encoding = PositionalEncoding(d_model, max_seq_len)
pe = pos_encoding.pe[0].numpy()  # [max_seq_len, d_model]

# Plot
plt.figure(figsize=(15, 5))
plt.imshow(pe.T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding Pattern')
plt.colorbar()
plt.show()

# Properties of sinusoidal encoding:
# 1. Unique encoding for each position
# 2. Encodes relative position (PE(pos+k) can be computed from PE(pos))
# 3. Generalizes to longer sequences than seen in training

# Alternative: Learned positional embeddings
class LearnedPositionalEmbedding(nn.Module):
    """
    Learn position embeddings (used in BERT, GPT)
    """
    
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.embedding(positions)

# Modern models (GPT-3, LLaMA) use learned embeddings
# Advantage: More flexible, can learn task-specific patterns
# Disadvantage: Fixed max length, doesn't generalize beyond training length
\`\`\`

---

## Feed-Forward Networks

### Position-wise FFN

\`\`\`python
"""
Position-wise feed-forward network
"""

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Applied independently to each position
    Same network at every position (but different across layers)
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern models use GELU instead of ReLU
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Expand to d_ff
        x = self.linear1(x)  # [batch, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project back to d_model
        x = self.linear2(x)  # [batch, seq_len, d_model]
        x = self.dropout(x)
        
        return x

# Typical sizes:
# d_model = 512, d_ff = 2048 (4x expansion)
# d_model = 768, d_ff = 3072 (4x expansion)
# d_model = 12288 (GPT-4 estimated), d_ff = 49152 (4x)

# Why FFN?
# 1. Add non-linearity (attention is linear transformation)
# 2. Position-wise: each position processed independently
# 3. Increases model capacity without increasing attention complexity

# Example
d_model = 512
d_ff = 2048
seq_len = 50

ffn = PositionwiseFeedForward(d_model, d_ff)

x = torch.randn(batch_size, seq_len, d_model)
output = ffn(x)

print(f"Input: {x.shape}")  # [32, 50, 512]
print(f"Output: {output.shape}")  # [32, 50, 512]
\`\`\`

---

## Complete Transformer Block

### Encoder Block

\`\`\`python
"""
Complete transformer encoder block
"""

class EncoderBlock(nn.Module):
    """
    Encoder block:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Sub-layer 1: Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Sub-layer 2: Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Sub-layer 1: Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        
        # Sub-layer 2: Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Add & Norm
        
        return x

# Why residual connections (x + sublayer(x))?
# 1. Gradient flow: Gradients can flow directly through addition
# 2. Easier optimization: Model can learn identity function if needed
# 3. Enables deeper networks: 100+ layers possible

# Why layer normalization?
# 1. Stabilizes training
# 2. Reduces internal covariate shift
# 3. Allows higher learning rates
\`\`\`

### Decoder Block

\`\`\`python
"""
Transformer decoder block with masked attention
"""

class DecoderBlock(nn.Module):
    """
    Decoder block:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Cross-attention with encoder output
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Sub-layer 1: Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Sub-layer 2: Cross-attention (decoder -> encoder)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Sub-layer 3: Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: decoder input [batch, tgt_len, d_model]
            encoder_output: encoder output [batch, src_len, d_model]
            src_mask: source mask [batch, 1, src_len]
            tgt_mask: target mask (causal) [batch, tgt_len, tgt_len]
        
        Returns:
            output: [batch, tgt_len, d_model]
        """
        # Sub-layer 1: Masked self-attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Sub-layer 2: Cross-attention
        cross_attn_output, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Sub-layer 3: Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x

def create_causal_mask(seq_len):
    """
    Create mask for autoregressive generation
    
    Prevents positions from attending to future positions
    Example for seq_len=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    
    Position i can only attend to positions <= i
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

# Usage
seq_len = 10
mask = create_causal_mask(seq_len)
print(mask[0, 0])
\`\`\`

---

## Complete Transformer Model

### Full Implementation

\`\`\`python
"""
Complete Transformer model for sequence-to-sequence tasks
"""

class Transformer(nn.Module):
    """
    Full Transformer: Encoder + Decoder
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights (Xavier uniform)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence
        
        Args:
            src: [batch, src_len]
            src_mask: [batch, 1, src_len]
        
        Returns:
            encoder_output: [batch, src_len, d_model]
        """
        # Embedding + positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence
        
        Args:
            tgt: [batch, tgt_len]
            encoder_output: [batch, src_len, d_model]
            src_mask: [batch, 1, src_len]
            tgt_mask: [batch, tgt_len, tgt_len]
        
        Returns:
            decoder_output: [batch, tgt_len, d_model]
        """
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass
        
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            src_mask: [batch, 1, src_len]
            tgt_mask: [batch, tgt_len, tgt_len]
        
        Returns:
            logits: [batch, tgt_len, tgt_vocab_size]
        """
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 10000
model = Transformer(src_vocab_size, tgt_vocab_size)

# Example batch
batch_size = 32
src_len = 20
tgt_len = 25

src = torch.randint(0, src_vocab_size, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

# Create masks
src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # Padding mask
tgt_mask = create_causal_mask(tgt_len).expand(batch_size, -1, -1, -1)

# Forward pass
logits = model(src, tgt, src_mask, tgt_mask)
print(f"Output shape: {logits.shape}")  # [32, 25, 10000]

# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~60M for this config
\`\`\`

---

## Decoder-Only Transformers (GPT)

### Simplified Architecture for Language Modeling

\`\`\`python
"""
Decoder-only transformer (GPT architecture)
"""

class GPTBlock(nn.Module):
    """
    GPT block: Simplified decoder (no cross-attention)
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Only self-attention (no cross-attention needed)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Masked self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class GPT(nn.Module):
    """
    GPT-style language model
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1
    ):
        super().__init__()
        
        # Token + position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer norm + output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share embeddings with output projection
        self.head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + position_emb)
        
        # Causal mask
        mask = create_causal_mask(seq_len).to(input_ids.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        return logits

# GPT-2 configurations
configs = {
    "gpt2-small": {"d_model": 768, "n_heads": 12, "n_layers": 12, "d_ff": 3072},
    "gpt2-medium": {"d_model": 1024, "n_heads": 16, "n_layers": 24, "d_ff": 4096},
    "gpt2-large": {"d_model": 1280, "n_heads": 20, "n_layers": 36, "d_ff": 5120},
    "gpt2-xl": {"d_model": 1600, "n_heads": 25, "n_layers": 48, "d_ff": 6400},
}

# Create GPT-2 small
model = GPT(vocab_size=50257, **configs["gpt2-small"])
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")  # ~124M
\`\`\`

---

## Conclusion

The Transformer architecture consists of:

1. **Multi-Head Attention**: Allows model to focus on different parts of input
2. **Positional Encodings**: Injects position information into embeddings
3. **Feed-Forward Networks**: Adds non-linearity and model capacity
4. **Residual Connections**: Enables deep networks with stable gradients
5. **Layer Normalization**: Stabilizes training

**Key Innovations**:
- Parallelization (vs sequential RNNs)
- Direct connections between any two positions
- Scalable to billions of parameters
- Foundation for all modern LLMs

**Practical Takeaways**:
- Use learned positional embeddings for language models
- Decoder-only (GPT) is simpler and works well for generation
- Multi-head attention learns different linguistic patterns
- Causal masking is critical for autoregressive generation
- Weight tying (input/output embeddings) reduces parameters

This architecture, with scale and data, produces GPT-4, Claude, and other modern LLMs.
`,
};
