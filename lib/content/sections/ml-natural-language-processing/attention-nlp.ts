/**
 * Section: Attention for NLP
 * Module: Natural Language Processing
 *
 * Covers attention mechanisms, self-attention, and multi-head attention for NLP
 */

export const attentionNlpSection = {
  id: 'attention-nlp',
  title: 'Attention for NLP',
  content: `
# Attention for NLP

## Introduction

Attention mechanisms revolutionized NLP by allowing models to focus on relevant parts of input when making predictions. Instead of compressing entire sequences into fixed-size vectors, attention dynamically weights different positions based on their relevance.

**Key Innovation:**
- **Dynamic focus**: Model learns what to attend to
- **Long-range dependencies**: Direct connections between any positions
- **Interpretability**: Attention weights show what model focuses on
- **Foundation for transformers**: Enabled BERT, GPT, and modern NLP

## The Attention Mechanism

### Intuition

\`\`\`
Translating: "I love machine learning" → "J'aime l'apprentissage automatique"

When generating "automatique":
- Should focus on: "machine" and "learning"
- Less relevant: "I", "love"

Attention computes: "How much should I focus on each input word?"
\`\`\`

### Mathematical Formulation

\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): What I'm looking for
- K (Key): What each position offers
- V (Value): Actual content at each position
- d_k: Dimension (scaling factor)
\`\`\`

### Attention Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention (nn.Module):
    """Scaled Dot-Product Attention"""
    
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        
    def forward (self, Q, K, V, mask=None):
        """
        Q: (batch, num_heads, seq_len, d_k)
        K: (batch, num_heads, seq_len, d_k)
        V: (batch, num_heads, seq_len, d_v)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k **0.5)
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided, for padding)
        if mask is not None:
            scores = scores.masked_fill (mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax (scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul (attention_weights, V)
        
        return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 32, 10, 512
d_k = d_v = 64

Q = torch.randn (batch_size, 1, seq_len, d_k)
K = torch.randn (batch_size, 1, seq_len, d_k)
V = torch.randn (batch_size, 1, seq_len, d_v)

attention = ScaledDotProductAttention (d_k)
output, weights = attention(Q, K, V)

print(f"Output shape: {output.shape}")  # (32, 1, 10, 64)
print(f"Attention weights shape: {weights.shape}")  # (32, 1, 10, 10)
\`\`\`

## Self-Attention

Self-attention computes attention within the same sequence - each position attends to all positions including itself.

\`\`\`python
class SelfAttention (nn.Module):
    """Self-Attention Layer"""
    
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear (d_model, d_k)
        self.W_k = nn.Linear (d_model, d_k)
        self.W_v = nn.Linear (d_k, d_k)
        
    def forward (self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # Project to Q, K, V
        Q = self.W_q (x)  # (batch, seq_len, d_k)
        K = self.W_k (x)  # (batch, seq_len, d_k)
        V = self.W_v (x)  # (batch, seq_len, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k **0.5)
        attention_weights = F.softmax (scores, dim=-1)
        
        # Apply attention
        output = torch.matmul (attention_weights, V)
        
        return output, attention_weights

# Example
d_model = 512
d_k = 64
seq_len = 20

x = torch.randn(32, seq_len, d_model)
self_attn = SelfAttention (d_model, d_k)
output, weights = self_attn (x)

print(f"Output shape: {output.shape}")  # (32, 20, 64)
print(f"Attention pattern: {weights[0, 0, :5]}")  # First 5 positions
\`\`\`

## Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to attend to different aspects simultaneously.

\`\`\`python
class MultiHeadAttention (nn.Module):
    """Multi-Head Attention"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear (d_model, d_model)
        self.W_k = nn.Linear (d_model, d_model)
        self.W_v = nn.Linear (d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear (d_model, d_model)
        
    def forward (self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)
        
        # Linear projections and reshape to (batch, num_heads, seq_len, d_k)
        Q = self.W_q(Q).view (batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view (batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view (batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k **0.5)
        
        if mask is not None:
            scores = scores.masked_fill (mask == 0, -1e9)
        
        attention_weights = F.softmax (scores, dim=-1)
        x = torch.matmul (attention_weights, V)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view (batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o (x)
        
        return output, attention_weights

# Example
d_model = 512
num_heads = 8
seq_len = 20

Q = K = V = torch.randn(32, seq_len, d_model)

mha = MultiHeadAttention (d_model, num_heads)
output, weights = mha(Q, K, V)

print(f"Output shape: {output.shape}")  # (32, 20, 512)
print(f"Attention weights shape: {weights.shape}")  # (32, 8, 20, 20)
\`\`\`

## Attention Visualization

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention (attention_weights, tokens, layer=0, head=0):
    """
    Visualize attention patterns
    attention_weights: (num_layers, num_heads, seq_len, seq_len)
    tokens: list of token strings
    """
    # Extract specific layer and head
    attn = attention_weights[layer, head].detach().cpu().numpy()
    
    plt.figure (figsize=(10, 8))
    sns.heatmap (attn, xticklabels=tokens, yticklabels=tokens, 
                cmap='viridis', cbar=True)
    plt.title (f'Attention Pattern (Layer {layer}, Head {head})')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.show()

# Example tokens
tokens = ['I', 'love', 'machine', 'learning']

# Simulate attention weights
attention = torch.randn(2, 8, 4, 4).softmax (dim=-1)

visualize_attention (attention, tokens, layer=0, head=0)
\`\`\`

## Attention Patterns in Practice

Different attention heads learn different patterns:

\`\`\`python
# Positional attention: focuses on adjacent words
# Head 1: Syntactic patterns (e.g., noun-verb relationships)
# Head 2: Semantic patterns (e.g., subject-object)
# Head 3: Long-range dependencies

# Example sentence: "The cat sat on the mat"
# When processing "sat":
# - Head 1 might focus on "cat" (subject)
# - Head 2 might focus on "on" (preposition)
# - Head 3 might focus on "mat" (object of preposition)
\`\`\`

## Masked Attention

For generation tasks, prevent attending to future tokens:

\`\`\`python
def create_causal_mask (seq_len):
    """Create causal mask for autoregressive generation"""
    mask = torch.tril (torch.ones (seq_len, seq_len))
    return mask

# Example
seq_len = 5
mask = create_causal_mask (seq_len)
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])

# Position i can only attend to positions ≤ i
# Prevents "cheating" by seeing future tokens
\`\`\`

## Attention vs Traditional Sequence Models

\`\`\`
LSTM:
- Sequential processing (slow)
- Information flows through hidden state (bottleneck)
- Indirect connections between distant positions

Attention:
- Parallel processing (fast)
- Direct connections between all positions
- Explicit attention weights (interpretable)
\`\`\`

## Why Attention Works

1. **Parallelization**: All positions processed simultaneously
2. **Long-range dependencies**: Direct connections, no distance decay
3. **Interpretability**: Can visualize what model attends to
4. **Flexibility**: Can attend to relevant information regardless of distance
5. **No vanishing gradients**: Direct paths for gradient flow

## Practical Applications

### Sentiment Analysis with Attention

\`\`\`python
class AttentionSentimentClassifier (nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding (vocab_size, embedding_dim)
        self.attention = MultiHeadAttention (embedding_dim, num_heads)
        self.fc = nn.Linear (embedding_dim, output_dim)
        
    def forward (self, x):
        # x: (batch, seq_len)
        embedded = self.embedding (x)  # (batch, seq_len, embedding_dim)
        
        # Self-attention
        attn_output, attn_weights = self.attention (embedded, embedded, embedded)
        
        # Average pooling
        pooled = attn_output.mean (dim=1)  # (batch, embedding_dim)
        
        # Classify
        logits = self.fc (pooled)
        
        return logits, attn_weights

# Can visualize which words the model focused on for classification!
\`\`\`

## Summary

Attention mechanisms:
- Allow models to focus on relevant information
- Enable parallel processing
- Provide interpretability
- Form foundation of transformers
- Revolutionized NLP

**Next**: We'll see how attention is used in transformer architecture (BERT, GPT).
`,
};
