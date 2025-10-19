/**
 * Recurrent Neural Networks (RNNs) Section
 */

export const recurrentNeuralNetworksSection = {
  id: 'recurrent-neural-networks',
  title: 'Recurrent Neural Networks (RNNs)',
  content: `# Recurrent Neural Networks (RNNs)

## Introduction

While CNNs excel at spatial data (images), many real-world problems involve **sequential data**: time series, text, audio, video. Recurrent Neural Networks (RNNs) are designed to process sequences by maintaining an internal state (memory) that captures information from previous timesteps.

**Sequential Data Examples**:
- **Text**: "The cat sat on the ___" (predict next word)
- **Time Series**: Stock prices, temperature, sensor readings
- **Audio**: Speech recognition, music generation
- **Video**: Activity recognition, video captioning
- **Biology**: DNA/protein sequences

**Why RNNs Matter**:
- Handle variable-length sequences
- Share parameters across time (like CNNs across space)
- Maintain temporal dependencies
- Foundation for modern sequence models

## The Problem with Feedforward Networks

### Why Not Use Regular Neural Networks?

**Problem 1: Fixed Input Size**
- Feedforward networks require fixed-size inputs
- Sequences have variable lengths
- Solution: Padding/truncation (wasteful and limiting)

**Problem 2: No Parameter Sharing**
- Learning at position 1 doesn't help position 100
- Must relearn same patterns for each position
- Huge parameter count

**Problem 3: No Temporal Memory**
- Each input processed independently
- Can't remember previous context
- "The clouds are in the ___" → need to remember "clouds" to predict "sky"

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Problem: Feedforward network for sequences
class FeedforwardSequence(nn.Module):
    """Naive approach: Flatten entire sequence."""
    def __init__(self, seq_length, input_size, hidden_size, output_size):
        super().__init__()
        # Must specify sequence length!
        self.fc1 = nn.Linear(seq_length * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_length, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten: loses temporal structure
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example problems
print("Feedforward Network Problems for Sequences:")
print("=" * 60)
print("\\n1. Fixed sequence length:")
print("   - seq_length=10: 10 × 50 = 500 input features")
print("   - seq_length=100: 100 × 50 = 5000 input features")
print("   - Different models for different lengths!")
print("\\n2. No parameter sharing:")
print("   - Learning 'cat' at position 1 ≠ position 50")
print("   - Parameters grow with sequence length")
print("\\n3. No temporal memory:")
print("   - All positions processed independently")
print("   - Can't maintain context across time")
\`\`\`

## RNN Basics

### Core Idea: Recurrence

**Key Insight**: Process sequence one step at a time, maintaining a **hidden state** that summarizes past information.

**RNN Formula**:
\`\`\`
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
\`\`\`

Where:
- \`x_t\`: Input at time t
- \`h_t\`: Hidden state at time t (memory)
- \`y_t\`: Output at time t
- \`W_hh\`: Hidden-to-hidden weights (recurrent)
- \`W_xh\`: Input-to-hidden weights
- \`W_hy\`: Hidden-to-output weights

**Visualization**:
\`\`\`
      y_0      y_1      y_2      y_3
       ↑        ↑        ↑        ↑
     [RNN] → [RNN] → [RNN] → [RNN]
       ↑    h_0 ↑    h_1 ↑    h_2 ↑
      x_0      x_1      x_2      x_3
\`\`\`

The same RNN cell (same weights) is applied at each timestep!

\`\`\`python
# RNN from scratch
class SimpleRNN:
    """
    Vanilla RNN implementation for understanding.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.hidden_size = hidden_size
        
        # Input to hidden
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        
        # Hidden to hidden (recurrent connection)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        
        # Hidden to output
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        """
        Forward pass through sequence.
        
        Args:
            inputs: List of inputs, each shape (input_size, 1)
        
        Returns:
            outputs: List of outputs
            hidden_states: List of hidden states
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        
        outputs = []
        hidden_states = [h]
        
        # Process sequence
        for x_t in inputs:
            # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
            h = np.tanh(
                np.dot(self.W_hh, h) + 
                np.dot(self.W_xh, x_t) + 
                self.b_h
            )
            
            # y_t = W_hy * h_t + b_y
            y = np.dot(self.W_hy, h) + self.b_y
            
            outputs.append(y)
            hidden_states.append(h)
        
        return outputs, hidden_states

# Example: Character-level RNN
print("\\nSimple RNN Example:")
print("=" * 60)

# Vocabulary
chars = ['h', 'e', 'l', 'o']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert 'hello' to one-hot vectors
def char_to_onehot(char):
    vec = np.zeros((len(chars), 1))
    vec[char_to_idx[char]] = 1
    return vec

text = 'hello'
inputs = [char_to_onehot(ch) for ch in text]

# Create and run RNN
rnn = SimpleRNN(input_size=len(chars), hidden_size=10, output_size=len(chars))
outputs, hidden_states = rnn.forward(inputs)

print(f"Input sequence: {text}")
print(f"Sequence length: {len(inputs)}")
print(f"Hidden size: {rnn.hidden_size}")
print(f"\\nHidden state shapes: {[h.shape for h in hidden_states[:3]]}")
print(f"Output shapes: {[y.shape for y in outputs[:3]]}")

# Visualize hidden states
hidden_matrix = np.hstack(hidden_states)
plt.figure(figsize=(12, 4))
plt.imshow(hidden_matrix, aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.xlabel('Time Step')
plt.ylabel('Hidden Unit')
plt.title('RNN Hidden State Evolution Over Time')
plt.show()

print("\\nKey Observations:")
print("✓ Same RNN weights used at each timestep")
print("✓ Hidden state carries information forward")
print("✓ Can process variable-length sequences")
print("✓ Parameter count independent of sequence length")
\`\`\`

## PyTorch RNN Implementation

\`\`\`python
# PyTorch RNN
class CharRNN(nn.Module):
    """
    Character-level RNN in PyTorch.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_length, input_size)
            hidden: Initial hidden state (optional)
        
        Returns:
            output: (batch, seq_length, output_size)
            hidden: Final hidden state
        """
        # RNN forward
        # out: (batch, seq, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        
        # Reshape for linear layer
        # out: (batch * seq, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        
        # Linear layer
        out = self.fc(out)
        
        # Reshape back
        # out: (batch, seq, output_size)
        batch_size = x.size(0)
        seq_length = x.size(1)
        out = out.view(batch_size, seq_length, -1)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Create model
vocab_size = 50
hidden_size = 128
model = CharRNN(vocab_size, hidden_size, vocab_size, num_layers=2)

print("PyTorch RNN Model:")
print(model)

# Test forward pass
batch_size = 32
seq_length = 20
x = torch.randn(batch_size, seq_length, vocab_size)

output, hidden = model(x)
print(f"\\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Hidden shape: {hidden.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total_params:,}")
\`\`\`

## Backpropagation Through Time (BPTT)

### Training RNNs

RNNs are trained using **Backpropagation Through Time (BPTT)**: unroll the network through time and apply standard backpropagation.

**Challenges**:
1. **Vanishing gradients**: Gradients shrink exponentially over long sequences
2. **Exploding gradients**: Gradients grow exponentially (less common)
3. **Computational cost**: Must store all intermediate states

\`\`\`python
# Training example: Character prediction
def train_char_rnn(model, data, epochs=10, seq_length=25):
    """
    Train character-level RNN.
    
    Args:
        model: CharRNN model
        data: Text data
        epochs: Number of training epochs
        seq_length: Length of training sequences
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert text to indices
    chars = sorted(list(set(data)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    
    for epoch in range(epochs):
        # Initialize hidden state
        hidden = model.init_hidden(batch_size=1)
        
        model.train()
        total_loss = 0
        
        # Create batches
        for i in range(0, len(data) - seq_length, seq_length):
            # Get sequence
            seq = data[i:i+seq_length]
            target = data[i+1:i+seq_length+1]
            
            # Convert to tensors
            seq_idx = torch.tensor([char_to_idx[ch] for ch in seq])
            target_idx = torch.tensor([char_to_idx[ch] for ch in target])
            
            # One-hot encode
            seq_onehot = F.one_hot(seq_idx, vocab_size).float().unsqueeze(0)
            
            # Forward pass
            optimizer.zero_grad()
            output, hidden = model(seq_onehot, hidden)
            
            # Detach hidden state (truncated BPTT)
            hidden = hidden.detach()
            
            # Compute loss
            loss = criterion(
                output.view(-1, vocab_size),
                target_idx.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(data) // seq_length)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Training Considerations:")
print("=" * 60)
print("\\n1. Truncated BPTT:")
print("   - Process long sequences in chunks")
print("   - Detach hidden state between chunks")
print("   - Reduces memory and computation")
print("\\n2. Gradient Clipping:")
print("   - Prevent exploding gradients")
print("   - Clip gradients to maximum norm")
print("   - Essential for stable training")
print("\\n3. Teacher Forcing:")
print("   - Use true previous output (not predicted)")
print("   - Faster convergence")
print("   - Can lead to exposure bias")
\`\`\`

## The Vanishing Gradient Problem

### Why RNNs Struggle with Long Sequences

**The Problem**:
- Gradient at time \`t\` depends on multiplication through all previous timesteps
- \`∂L/∂h_0 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_1/∂h_0\`
- If derivatives < 1, gradients **vanish** (→ 0) exponentially
- If derivatives > 1, gradients **explode** (→ ∞) exponentially

**Consequence**: RNNs can't learn long-term dependencies (50+ steps)

\`\`\`python
# Demonstrate vanishing gradients
def show_vanishing_gradients():
    """Visualize how gradients vanish over time."""
    # Simulate gradient flow
    seq_length = 100
    derivative = 0.9  # Typical value for tanh
    
    gradients = []
    grad = 1.0
    
    for t in range(seq_length):
        grad *= derivative
        gradients.append(grad)
    
    plt.figure(figsize=(12, 5))
    plt.semilogy(gradients)
    plt.xlabel('Timesteps Back')
    plt.ylabel('Gradient Magnitude (log scale)')
    plt.title('Vanishing Gradient in RNN')
    plt.grid(True)
    plt.show()
    
    print(f"\\nGradient after 10 steps: {gradients[10]:.6f}")
    print(f"Gradient after 50 steps: {gradients[50]:.2e}")
    print(f"Gradient after 100 steps: {gradients[99]:.2e}")
    print("\\n→ Gradients effectively zero after ~50 steps!")
    print("→ RNN can't learn dependencies longer than 50 steps")

show_vanishing_gradients()

print("\\n" + "=" * 60)
print("Solutions to Vanishing Gradients:")
print("=" * 60)
print("\\n1. LSTM (Long Short-Term Memory)")
print("   - Gating mechanisms")
print("   - Constant error carousel")
print("   - Next section!")
print("\\n2. GRU (Gated Recurrent Unit)")
print("   - Simplified LSTM")
print("   - Fewer parameters")
print("\\n3. Gradient Clipping")
print("   - Helps with exploding (not vanishing)")
print("\\n4. Better Initialization")
print("   - Identity initialization for W_hh")
print("   - ReLU activation (but has issues)")
print("\\n5. Skip Connections")
print("   - Residual connections in RNNs")
\`\`\`

## RNN Architectures

### Different RNN Patterns

**1. One-to-Many**: Single input → sequence output
- Example: Image captioning (image → sentence)

**2. Many-to-One**: Sequence input → single output  
- Example: Sentiment analysis (sentence → positive/negative)

**3. Many-to-Many (synced)**: Sequence → sequence (aligned)
- Example: Video classification (frame-by-frame labels)

**4. Many-to-Many (encoder-decoder)**: Sequence → sequence (unaligned)
- Example: Translation (English → French)

\`\`\`python
# Example: Sentiment Analysis (Many-to-One)
class SentimentRNN(nn.Module):
    """Many-to-one RNN for sentiment classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(SentimentRNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        
        # Classification layer (uses final hidden state)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_length) - word indices
        
        # Embed words
        embedded = self.embedding(x)  # (batch, seq, embedding_dim)
        
        # RNN forward
        output, hidden = self.rnn(embedded)
        # output: (batch, seq, hidden_size)
        # hidden: (1, batch, hidden_size)
        
        # Use final hidden state for classification
        final_hidden = hidden.squeeze(0)  # (batch, hidden_size)
        
        # Classify
        logits = self.fc(final_hidden)  # (batch, num_classes)
        
        return logits

# Create sentiment model
sentiment_model = SentimentRNN(
    vocab_size=10000,
    embedding_dim=100,
    hidden_size=256,
    num_classes=2  # Positive/Negative
)

print("Sentiment Analysis RNN:")
print(sentiment_model)

# Test
reviews = torch.randint(0, 10000, (32, 50))  # 32 reviews, 50 words each
output = sentiment_model(reviews)
print(f"\\nInput shape: {reviews.shape} (batch, sequence)")
print(f"Output shape: {output.shape} (batch, classes)")
print("\\n→ Takes variable-length sequences")
print("→ Outputs single sentiment prediction per review")
\`\`\`

## Key Takeaways

1. **RNNs process sequences** by maintaining hidden state (memory)

2. **Key advantages**:
   - Handle variable-length sequences
   - Parameter sharing across time
   - Model temporal dependencies

3. **RNN formula**: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

4. **Vanishing gradient problem**: RNNs struggle with long sequences (>50 steps)

5. **Solutions**: LSTM and GRU (next sections) solve vanishing gradients

6. **Architecture patterns**:
   - One-to-many, many-to-one
   - Many-to-many (synced and encoder-decoder)

7. **Training techniques**:
   - Truncated BPTT
   - Gradient clipping
   - Teacher forcing

## Coming Next

In the next section, we'll explore **LSTM & GRU** - advanced RNN architectures that solve the vanishing gradient problem and enable learning long-term dependencies!
`,
};
