/**
 * Attention Mechanism Section
 */

export const attentionMechanismSection = {
  id: 'attention-mechanism',
  title: 'Attention Mechanism',
  content: `# Attention Mechanism

## Introduction

The attention mechanism was a breakthrough that solved the Seq2Seq bottleneck problem and revolutionized deep learning. Instead of compressing the entire input into a fixed-size vector, attention allows the decoder to dynamically "attend" to different parts of the input at each generation step.

**The Breakthrough (2015)**:
- Paper: "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al.)
- Idea: Let decoder access all encoder hidden states, not just final one
- Impact: Massive improvement in translation quality, especially for long sequences

**Key Applications**:
- Machine translation (original application)
- Text summarization
- Image captioning
- Question answering
- Foundation for Transformers (next section)

## The Core Problem Revisited

### What Attention Solves

**Without Attention (Basic Seq2Seq)**:
\`\`\`
Encoder outputs: h_1, h_2, h_3, ..., h_n
                    ↓     ↓     ↓          ↓
                    └─────┴─────┴──────────┴→ Context c = h_n (final state only!)
                                                      ↓
                                              Decoder uses ONLY c
\`\`\`

**With Attention**:
\`\`\`
Encoder outputs: h_1, h_2, h_3, ..., h_n
                  ↓     ↓     ↓          ↓
                  ↓     ↓     ↓          ↓  All states preserved!
                  └─────┴─────┴──────────┘
                          ↓
                  Attention mechanism (dynamic weights)
                          ↓
                  Context c_t (different at each decoder step!)
                          ↓
                      Decoder
\`\`\`

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Visualize attention concept
def visualize_attention_concept():
    """Show how attention focuses on different input positions."""
    
    # Example: Translating "I love machine learning" to French
    source_words = ['I', 'love', 'machine', 'learning']
    target_words = ['J', 'aime', 'apprentissage', 'automatique']
    
    # Simulated attention weights (rows = target, cols = source)
    attention_weights = np.array([
        [0.8, 0.1, 0.05, 0.05],  # "J'" attends to "I"
        [0.1, 0.8, 0.05, 0.05],  # "aime" attends to "love"
        [0.0, 0.1, 0.7, 0.2],    # "apprentissage" attends to "machine"+"learning"
        [0.0, 0.0, 0.3, 0.7],    # "automatique" attends to "learning"
    ])
    
    # Plot attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    # Labels
    plt.xlabel('Source (English)', fontsize=12)
    plt.ylabel('Target (French)', fontsize=12)
    plt.xticks(range(len(source_words)), source_words)
    plt.yticks(range(len(target_words)), target_words)
    plt.title('Attention Weights: English → French Translation', fontsize=14)
    
    # Add text annotations
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Attention Mechanism:")
    print("=" * 60)
    print("When generating 'J':")
    print("  → Focuses on 'I' (0.8 weight)")
    print("\\nWhen generating 'aime':")
    print("  → Focuses on 'love' (0.8 weight)")
    print("\\nWhen generating 'apprentissage':")
    print("  → Focuses on 'machine' (0.7) and 'learning' (0.2)")
    print("\\nKey insight: Different parts of input relevant for different outputs!")

visualize_attention_concept()
\`\`\`

## Attention Mechanism: Step-by-Step

### The Three Core Steps

**Step 1: Score** - How relevant is each encoder state?
**Step 2: Align** - Convert scores to weights (softmax)
**Step 3: Context** - Weighted sum of encoder states

### Mathematical Formulation

At decoder timestep t:

1. **Score Function** (Bahdanau/Additive):
\`\`\`
score(h_i, s_{t-1}) = v^T tanh(W_1 h_i + W_2 s_{t-1})
\`\`\`
where:
- h_i = encoder hidden state at position i
- s_{t-1} = decoder hidden state at previous timestep
- W_1, W_2, v = learnable parameters

2. **Alignment Weights** (Softmax):
\`\`\`
α_{t,i} = exp(score(h_i, s_{t-1})) / Σ_j exp(score(h_j, s_{t-1}))
\`\`\`
Ensures weights sum to 1: Σ_i α_{t,i} = 1

3. **Context Vector** (Weighted Sum):
\`\`\`
c_t = Σ_i α_{t,i} * h_i
\`\`\`

4. **Decoder Update**:
\`\`\`
s_t = f(s_{t-1}, y_{t-1}, c_t)
output_t = g(s_t, c_t)
\`\`\`

\`\`\`python
class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Attention parameters
        self.W_h = nn.Linear(hidden_size, hidden_size)  # For encoder hidden states
        self.W_s = nn.Linear(hidden_size, hidden_size)  # For decoder hidden state
        self.v = nn.Linear(hidden_size, 1)              # Final projection
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Compute attention.
        
        Args:
            encoder_outputs: (batch, seq_length, hidden_size)
            decoder_hidden: (batch, hidden_size)
        
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_length)
        """
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        # Expand decoder hidden to match encoder sequence length
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_length, 1)
        # decoder_hidden: (batch, seq_length, hidden_size)
        
        # Compute scores (energy)
        # score = v^T * tanh(W_h * h_i + W_s * s_{t-1})
        energy = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(decoder_hidden)
        )
        # energy: (batch, seq_length, hidden_size)
        
        # Project to scalar scores
        scores = self.v(energy).squeeze(2)
        # scores: (batch, seq_length)
        
        # Compute attention weights (alignment)
        attention_weights = F.softmax(scores, dim=1)
        # attention_weights: (batch, seq_length)
        
        # Compute context vector (weighted sum)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_length)
            encoder_outputs                  # (batch, seq_length, hidden_size)
        ).squeeze(1)
        # context: (batch, hidden_size)
        
        return context, attention_weights

# Test attention mechanism
print("Attention Mechanism Implementation:")
print("=" * 60)

hidden_size = 256
batch_size = 2
seq_length = 10

attention = BahdanauAttention(hidden_size)

# Dummy encoder outputs and decoder hidden state
encoder_outputs = torch.randn(batch_size, seq_length, hidden_size)
decoder_hidden = torch.randn(batch_size, hidden_size)

# Compute attention
context, weights = attention(encoder_outputs, decoder_hidden)

print(f"Encoder outputs shape: {encoder_outputs.shape}")
print(f"Decoder hidden shape: {decoder_hidden.shape}")
print(f"\\nContext vector shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\\nAttention weights for first sample:")
print(weights[0].detach().numpy())
print(f"Sum of weights: {weights[0].sum().item():.4f} (should be 1.0)")

# Visualize attention weights
plt.figure(figsize=(12, 4))
plt.bar(range(seq_length), weights[0].detach().numpy())
plt.xlabel('Encoder Position')
plt.ylabel('Attention Weight')
plt.title('Attention Distribution Over Input Sequence')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Attention Variants

### Different Score Functions

**1. Dot Product (Luong)**:
\`\`\`
score(h_i, s_t) = h_i^T s_t
\`\`\`
- Simplest
- No parameters
- Requires same dimensionality

**2. General (Luong)**:
\`\`\`
score(h_i, s_t) = h_i^T W s_t
\`\`\`
- Learns weight matrix W
- Can handle different dimensions

**3. Concat/Additive (Bahdanau)**:
\`\`\`
score(h_i, s_t) = v^T tanh(W_1 h_i + W_2 s_t)
\`\`\`
- Most parameters
- Often best performance

\`\`\`python
class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism.
    """
    def __init__(self, hidden_size, method='general'):
        super(LuongAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Compute Luong attention.
        
        Args:
            encoder_outputs: (batch, seq_length, hidden_size)
            decoder_hidden: (batch, hidden_size)
        
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_length)
        """
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)
        
        if self.method == 'dot':
            # score(h_i, s_t) = h_i^T · s_t
            scores = torch.bmm(
                encoder_outputs,                     # (batch, seq, hidden)
                decoder_hidden.unsqueeze(2)          # (batch, hidden, 1)
            ).squeeze(2)
            # scores: (batch, seq_length)
        
        elif self.method == 'general':
            # score(h_i, s_t) = h_i^T · W · s_t
            transformed = self.W(decoder_hidden)     # (batch, hidden)
            scores = torch.bmm(
                encoder_outputs,                     # (batch, seq, hidden)
                transformed.unsqueeze(2)             # (batch, hidden, 1)
            ).squeeze(2)
        
        elif self.method == 'concat':
            # score(h_i, s_t) = v^T · tanh(W · [h_i; s_t])
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_length, 1)
            concat = torch.cat([encoder_outputs, decoder_hidden], dim=2)
            energy = torch.tanh(self.W(concat))
            scores = self.v(energy).squeeze(2)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        
        return context, attention_weights

# Compare attention mechanisms
print("\\nAttention Mechanism Comparison:")
print("=" * 60)

mechanisms = {
    'Dot Product': LuongAttention(hidden_size, method='dot'),
    'General': LuongAttention(hidden_size, method='general'),
    'Concat': LuongAttention(hidden_size, method='concat'),
}

for name, attn in mechanisms.items():
    params = sum(p.numel() for p in attn.parameters())
    print(f"{name:15s}: {params:,} parameters")

print("\\nTrade-offs:")
print("  Dot Product: Fastest, no params, requires same dimensions")
print("  General: Good balance, moderate params")
print("  Concat: Most expressive, most params, often best quality")
\`\`\`

## Encoder-Decoder with Attention

### Complete Architecture

\`\`\`python
class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism.
    """
    def __init__(self, output_size, embedding_dim, hidden_size, attention_type='bahdanau'):
        super(AttentionDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Embedding
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        else:
            self.attention = LuongAttention(hidden_size, method='general')
        
        # LSTM (input = embedding + context)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, batch_first=True)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        Decode one step with attention.
        
        Args:
            input_token: (batch, 1) - current input token
            hidden: (1, batch, hidden_size) - previous hidden state
            cell: (1, batch, hidden_size) - previous cell state
            encoder_outputs: (batch, seq_length, hidden_size) - all encoder states
        
        Returns:
            output: (batch, output_size) - predicted token logits
            hidden: Updated hidden state
            cell: Updated cell state
            attention_weights: (batch, seq_length)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch, 1, embedding_dim)
        
        # Compute attention
        # hidden: (1, batch, hidden_size) → squeeze to (batch, hidden_size)
        context, attention_weights = self.attention(encoder_outputs, hidden.squeeze(0))
        # context: (batch, hidden_size)
        
        # Concatenate embedding and context
        context = context.unsqueeze(1)  # (batch, 1, hidden_size)
        lstm_input = torch.cat([embedded, context], dim=2)
        # lstm_input: (batch, 1, embedding_dim + hidden_size)
        
        # LSTM step
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch, 1, hidden_size)
        
        # Project to vocabulary
        output = self.fc(output.squeeze(1))  # (batch, output_size)
        
        return output, hidden, cell, attention_weights

class Seq2SeqWithAttention(nn.Module):
    """
    Complete Seq2Seq model with attention.
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        Forward pass through Seq2Seq with attention.
        
        Args:
            source: (batch, source_length)
            target: (batch, target_length)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: (batch, target_length, output_size)
            attention_history: List of attention weights at each step
        """
        batch_size = source.size(0)
        target_length = target.size(1)
        output_size = self.decoder.output_size
        
        # Store outputs and attention
        outputs = torch.zeros(batch_size, target_length, output_size).to(self.device)
        attention_history = []
        
        # Encode entire input sequence
        encoder_outputs, (hidden, cell) = self.encoder(source)
        # encoder_outputs: (batch, source_length, hidden_size)
        
        # First input to decoder
        decoder_input = target[:, 0].unsqueeze(1)  # <START> token
        
        # Decode with attention
        for t in range(1, target_length):
            # Decode one step
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            # Store output and attention
            outputs[:, t, :] = output
            attention_history.append(attention_weights.detach())
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs, attention_history

print("\\nSeq2Seq with Attention:")
print("=" * 60)

from types import SimpleNamespace

# Create simple encoder (reuse from previous)
encoder = Encoder(input_size=5000, embedding_dim=256, hidden_size=512)
decoder = AttentionDecoder(output_size=5000, embedding_dim=256, hidden_size=512)

device = torch.device('cpu')
model = Seq2SeqWithAttention(encoder, decoder, device)

# Test
source = torch.randint(0, 5000, (2, 15))  # Batch of 2, source length 15
target = torch.randint(0, 5000, (2, 20))  # Batch of 2, target length 20

outputs, attention_history = model(source, target, teacher_forcing_ratio=0.5)

print(f"Source shape: {source.shape}")
print(f"Target shape: {target.shape}")
print(f"Output shape: {outputs.shape}")
print(f"\\nAttention history: {len(attention_history)} timesteps")
print(f"Each attention: shape {attention_history[0].shape}")
print("\\n→ Decoder can attend to any encoder position!")
print("→ No more bottleneck!")
\`\`\`

## Visualizing Attention

### Attention Heatmaps

Attention weights reveal what the model is "looking at" when generating each output token.

\`\`\`python
def visualize_attention_heatmap(source_tokens, target_tokens, attention_weights):
    """
    Visualize attention as a heatmap.
    
    Args:
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        attention_weights: (target_len, source_len) array
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    plt.imshow(attention_weights, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    
    # Set ticks and labels
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=45, ha='right')
    plt.yticks(range(len(target_tokens)), target_tokens)
    
    plt.xlabel('Source (Input)', fontsize=12)
    plt.ylabel('Target (Output)', fontsize=12)
    plt.title('Attention Heatmap: Translation', fontsize=14)
    
    # Add grid
    plt.grid(False)
    
    # Add values as text
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            value = attention_weights[i, j]
            if value > 0.1:  # Only show significant weights
                color = 'white' if value > 0.5 else 'black'
                plt.text(j, i, f'{value:.2f}', 
                        ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Example: Attention for translation
source = "The cat sat on the mat".split()
target = "Le chat était assis sur le tapis".split()

# Simulated attention weights
attention = np.array([
    [0.7, 0.2, 0.05, 0.05, 0.0, 0.0],   # "Le" → "The"
    [0.1, 0.8, 0.05, 0.05, 0.0, 0.0],   # "chat" → "cat"
    [0.0, 0.2, 0.7, 0.1, 0.0, 0.0],     # "était" → "sat"
    [0.0, 0.1, 0.7, 0.2, 0.0, 0.0],     # "assis" → "sat"
    [0.0, 0.0, 0.1, 0.6, 0.2, 0.1],     # "sur" → "on"
    [0.05, 0.05, 0.0, 0.2, 0.6, 0.1],   # "le" → "the"
    [0.0, 0.0, 0.0, 0.1, 0.2, 0.7],     # "tapis" → "mat"
])

visualize_attention_heatmap(source, target, attention)

print("\\nAttention Visualization Insights:")
print("=" * 60)
print("Diagonal patterns: Word-by-word alignment")
print("Vertical bands: One source word affects multiple target words")
print("Horizontal bands: Multiple source words affect one target word")
print("\\nAttention is interpretable - we can see what model is focusing on!")
\`\`\`

## Benefits of Attention

### Key Advantages

1. **Solves Bottleneck**: No fixed-size compression
2. **Long Sequences**: Performance doesn't degrade with length
3. **Interpretability**: Visualize what model focuses on
4. **Better Gradients**: Direct path to relevant inputs
5. **Alignment**: Model learns input-output alignment

\`\`\`python
# Compare performance: with vs without attention
print("\\nPerformance Comparison (Typical):")
print("=" * 60)
print("\\nMachine Translation Quality (BLEU score):")
print("\\nSequence Length | Without Attention | With Attention | Improvement")
print("-" * 60)

data = [
    (10, 35.2, 36.5, "+1.3"),
    (20, 30.1, 33.8, "+3.7"),
    (30, 24.5, 32.1, "+7.6"),
    (40, 19.2, 30.5, "+11.3"),
    (50, 15.1, 29.2, "+14.1"),
]

for length, without, with_attn, improvement in data:
    print(f"{length:3d} words     | {without:5.1f}            | {with_attn:5.1f}          | {improvement}")

print("\\nKey Observation:")
print("→ Without attention: Significant degradation on long sequences")
print("→ With attention: Consistent performance across all lengths!")
print("→ Improvement increases with sequence length")
\`\`\`

## Key Takeaways

1. **Attention solves the bottleneck problem** by allowing decoder to access all encoder states

2. **Three steps**: Score, Align (softmax), Context (weighted sum)

3. **Multiple variants**: Bahdanau (additive), Luong (multiplicative), Dot product

4. **Context vector c_t**: Different at each decoder timestep, computed dynamically

5. **Interpretability**: Attention weights show what model focuses on

6. **Performance**: Dramatically improves long sequence handling

7. **Foundation for Transformers**: Self-attention extends this idea

8. **Universal mechanism**: Now used across all deep learning (vision, speech, etc.)

## Coming Next

In the next section, we'll explore the **Transformer Architecture** - which uses self-attention exclusively and revolutionized NLP, leading to BERT, GPT, and modern LLMs!
`,
};
