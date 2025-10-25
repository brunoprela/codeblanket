/**
 * LSTM & GRU Section
 */

export const lstmGruSection = {
  id: 'lstm-gru',
  title: 'LSTM & GRU',
  content: `# LSTM & GRU

## Introduction

Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) are advanced RNN architectures that solve the vanishing gradient problem, enabling learning of long-term dependencies. They've become the standard for sequence modeling tasks.

**Why LSTM/GRU Matter**:
- Solve vanishing gradient problem
- Learn dependencies 100+ timesteps away
- State-of-the-art for many sequence tasks
- Foundation for modern NLP (before Transformers)

**Key Innovation**: **Gates** - learnable mechanisms that control information flow, allowing gradients to flow unchanged over many timesteps.

## The LSTM Architecture

### Core Idea: Cell State

LSTM introduces a **cell state** (C_t) - a "conveyor belt" that runs through the sequence with minimal modifications, allowing information to flow unchanged over long distances.

**LSTM Components**:
1. **Cell State (C_t)**: Long-term memory
2. **Hidden State (h_t)**: Short-term memory (like vanilla RNN)
3. **Three Gates**: Control information flow
   - Forget gate: What to remove from memory
   - Input gate: What new information to add
   - Output gate: What to output

### LSTM Equations

\`\`\`
Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:    C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Cell Update:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State: h_t = o_t ⊙ tanh(C_t)
\`\`\`

Where:
- σ = sigmoid (outputs 0-1, gate activation)
- ⊙ = element-wise multiplication
- tanh = hyperbolic tangent
- [h_{t-1}, x_t] = concatenation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# LSTM from scratch for understanding
class LSTMCell:
    """
    Single LSTM cell implementation.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for all gates
        combined_size = hidden_size + input_size
        
        # Forget gate
        self.W_f = np.random.randn (hidden_size, combined_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Input gate
        self.W_i = np.random.randn (hidden_size, combined_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Candidate gate
        self.W_c = np.random.randn (hidden_size, combined_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_o = np.random.randn (hidden_size, combined_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
    
    def sigmoid (self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward (self, x_t, h_prev, c_prev):
        """
        Forward pass through LSTM cell.
        
        Args:
            x_t: Input at time t, shape (input_size, 1)
            h_prev: Previous hidden state, shape (hidden_size, 1)
            c_prev: Previous cell state, shape (hidden_size, 1)
        
        Returns:
            h_t: New hidden state
            c_t: New cell state
        """
        # Concatenate previous hidden state and current input
        combined = np.vstack([h_prev, x_t])  # (hidden_size + input_size, 1)
        
        # Forget gate: What to forget from cell state
        f_t = self.sigmoid (np.dot (self.W_f, combined) + self.b_f)
        
        # Input gate: What new information to store
        i_t = self.sigmoid (np.dot (self.W_i, combined) + self.b_i)
        
        # Candidate cell state: New candidate values
        c_tilde = np.tanh (np.dot (self.W_c, combined) + self.b_c)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate: What to output
        o_t = self.sigmoid (np.dot (self.W_o, combined) + self.b_o)
        
        # New hidden state
        h_t = o_t * np.tanh (c_t)
        
        return h_t, c_t, (f_t, i_t, c_tilde, o_t)

# Demonstrate LSTM cell
print("LSTM Cell Demonstration:")
print("=" * 60)

input_size = 10
hidden_size = 20
lstm_cell = LSTMCell (input_size, hidden_size)

# Initial states
h_0 = np.zeros((hidden_size, 1))
c_0 = np.zeros((hidden_size, 1))

# Process sequence
x = np.random.randn (input_size, 1)
h_1, c_1, gates = lstm_cell.forward (x, h_0, c_0)

f_t, i_t, c_tilde, o_t = gates

print(f"Input shape: {x.shape}")
print(f"Hidden state shape: {h_1.shape}")
print(f"Cell state shape: {c_1.shape}")
print(f"\\nGate Activations (first 5 values):")
print(f"  Forget gate (f_t): {f_t[:5, 0].T}")
print(f"  Input gate (i_t): {i_t[:5, 0].T}")
print(f"  Output gate (o_t): {o_t[:5, 0].T}")

print("\\nGate Interpretation:")
print("  f_t ≈ 1 → Keep cell state")
print("  f_t ≈ 0 → Forget cell state")
print("  i_t ≈ 1 → Accept new information")
print("  i_t ≈ 0 → Ignore new information")
print("  o_t ≈ 1 → Output cell state")
print("  o_t ≈ 0 → Hide cell state")
\`\`\`

## Understanding the Gates

### 1. Forget Gate

**Purpose**: Decide what information to throw away from cell state

**Mechanism**: 
\`\`\`
f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
\`\`\`

- Looks at h_{t-1} and x_t
- Outputs value between 0 and 1 for each number in cell state
- f_t = 1: "Keep this completely"
- f_t = 0: "Forget this completely"

**Example**: "John is tall. He ___"
- After seeing "He", forget gate might forget "John" from cell state
- Later: "Mary is short. She ___"  
- After "She", forget "Mary", reset for new subject

### 2. Input Gate

**Purpose**: Decide what new information to store in cell state

**Mechanism**:
\`\`\`
i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
\`\`\`

- Input gate (i_t): How much of candidate to add?
- Candidate (C̃_t): New candidate values to add
- Both look at h_{t-1} and x_t

**Example**: "The cat, which was orange, ___"
- Input gate decides to store "cat" (subject) in cell state
- Candidate creates representation of "cat"

### 3. Cell State Update

**Purpose**: Actually update the cell state

**Mechanism**:
\`\`\`
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
      ↑             ↑
   Forget old    Add new
\`\`\`

- Multiply old cell state by forget gate (forget old info)
- Add new candidate values scaled by input gate (add new info)

### 4. Output Gate

**Purpose**: Decide what to output based on cell state

**Mechanism**:
\`\`\`
o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
\`\`\`

- Output gate decides what parts of cell state to output
- tanh(C_t): squash cell state to [-1, 1]
- Multiply by output gate to filter

\`\`\`python
# Visualize gate behavior
def visualize_lstm_gates (sequence_length=50):
    """Visualize how LSTM gates evolve over time."""
    
    # Simulate gate activations
    np.random.seed(42)
    forget_gate = np.random.beta(2, 2, sequence_length) # Tends toward 0.5
    input_gate = np.random.beta(2, 5, sequence_length)  # Tends toward 0
    output_gate = np.random.beta(3, 2, sequence_length) # Tends toward 0.6
    
    # Cell state evolution
    cell_state = np.zeros (sequence_length)
    cell_state[0] = 0.5
    
    for t in range(1, sequence_length):
        # Simplified: C_t = f_t * C_{t-1} + i_t * input
        cell_state[t] = forget_gate[t] * cell_state[t-1] + input_gate[t] * 0.5
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Forget gate
    axes[0].plot (forget_gate, color='red', linewidth=2)
    axes[0].fill_between (range (sequence_length), 0, forget_gate, alpha=0.3, color='red')
    axes[0].set_ylabel('Forget Gate', fontsize=12)
    axes[0].set_title('LSTM Gate Activations Over Time', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Input gate
    axes[1].plot (input_gate, color='green', linewidth=2)
    axes[1].fill_between (range (sequence_length), 0, input_gate, alpha=0.3, color='green')
    axes[1].set_ylabel('Input Gate', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Output gate
    axes[2].plot (output_gate, color='blue', linewidth=2)
    axes[2].fill_between (range (sequence_length), 0, output_gate, alpha=0.3, color='blue')
    axes[2].set_ylabel('Output Gate', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    # Cell state
    axes[3].plot (cell_state, color='purple', linewidth=2)
    axes[3].fill_between (range (sequence_length), 0, cell_state, alpha=0.3, color='purple')
    axes[3].set_ylabel('Cell State', fontsize=12)
    axes[3].set_xlabel('Timestep', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_lstm_gates()

print("\\nGate Behavior Patterns:")
print("=" * 60)
print("\\nForget Gate (Red):")
print("  → High values = Retain information")
print("  → Used to maintain long-term memory")
print("\\nInput Gate (Green):")
print("  → Spikes = New important information")
print("  → Controls what gets added to memory")
print("\\nOutput Gate (Blue):")
print("  → Controls what information is exposed")
print("  → Can hide cell state when not relevant")
print("\\nCell State (Purple):")
print("  → Gradually accumulates information")
print("  → Persists over long sequences")
\`\`\`

## PyTorch LSTM

\`\`\`python
# LSTM in PyTorch
class CharacterLSTM(nn.Module):
    """
    Character-level language model with LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(CharacterLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding (vocab_size, embedding_dim)
        
        # LSTM layer (s)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear (hidden_size, vocab_size)
    
    def forward (self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_length) - character indices
            hidden: Tuple of (h_0, c_0) or None
        
        Returns:
            output: (batch, seq_length, vocab_size)
            hidden: Tuple of (h_n, c_n)
        """
        # Embed characters
        embedded = self.embedding (x)  # (batch, seq, embedding_dim)
        
        # LSTM forward
        lstm_out, hidden = self.lstm (embedded, hidden)
        # lstm_out: (batch, seq, hidden_size)
        # hidden: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
        
        # Reshape for linear layer
        batch_size, seq_length, hidden_size = lstm_out.shape
        lstm_out = lstm_out.contiguous().view(-1, hidden_size)
        
        # Output layer
        output = self.fc (lstm_out)  # (batch * seq, vocab_size)
        
        # Reshape back
        output = output.view (batch_size, seq_length, -1)
        
        return output, hidden
    
    def init_hidden (self, batch_size, device='cpu'):
        """Initialize hidden and cell states."""
        h_0 = torch.zeros (self.num_layers, batch_size, self.hidden_size).to (device)
        c_0 = torch.zeros (self.num_layers, batch_size, self.hidden_size).to (device)
        return (h_0, c_0)

# Create LSTM model
vocab_size = 100
embedding_dim = 64
hidden_size = 128
num_layers = 2

model = CharacterLSTM(vocab_size, embedding_dim, hidden_size, num_layers)

print("PyTorch LSTM Model:")
print(model)

# Count parameters
def count_params (model):
    return sum (p.numel() for p in model.parameters() if p.requires_grad)

print(f"\\nTotal parameters: {count_params (model):,}")

# Test forward pass
batch_size = 32
seq_length = 20
x = torch.randint(0, vocab_size, (batch_size, seq_length))

output, (h_n, c_n) = model (x)

print(f"\\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Final hidden state shape: {h_n.shape}")
print(f"Final cell state shape: {c_n.shape}")

# Parameter breakdown
lstm_params = sum (p.numel() for p in model.lstm.parameters())
embedding_params = sum (p.numel() for p in model.embedding.parameters())
fc_params = sum (p.numel() for p in model.fc.parameters())

print(f"\\nParameter Breakdown:")
print(f"  Embedding: {embedding_params:,}")
print(f"  LSTM: {lstm_params:,}")
print(f"  Output FC: {fc_params:,}")
\`\`\`

## Why LSTM Solves Vanishing Gradients

### The Gradient Highway

**Key Insight**: Cell state provides a **direct path** for gradients to flow backward through time with minimal modification.

**Gradient Flow**:
\`\`\`
∂C_t/∂C_{t-1} = f_t  (forget gate, typically 0.5-1.0)
\`\`\`

Compare to vanilla RNN:
\`\`\`
∂h_t/∂h_{t-1} = W_hh × diag (tanh'(...))  (many matrices, <1 values)
\`\`\`

**LSTM Advantage**:
- Cell state update is mostly **additive**: C_t = f_t ⊙ C_{t-1} + ...
- Gradients flow through addition (derivative = 1) not multiplication
- Forget gate controls flow, typically stays near 1 for important info
- Result: Gradients don't vanish over 100+ timesteps!

\`\`\`python
# Compare gradient flow: RNN vs LSTM
def compare_gradient_flow():
    """Compare how gradients decay in RNN vs LSTM."""
    
    sequence_length = 100
    
    # RNN: Gradient multiplied by derivative at each step
    rnn_derivative = 0.25  # Typical tanh derivative
    rnn_gradients = [(rnn_derivative ** t) for t in range (sequence_length)]
    
    # LSTM: Gradient mostly preserved through cell state
    lstm_forget_gate = 0.95  # Typical forget gate value
    lstm_gradients = [(lstm_forget_gate ** t) for t in range (sequence_length)]
    
    # Plot
    plt.figure (figsize=(14, 6))
    
    plt.semilogy (rnn_gradients, label='RNN (vanilla)', linewidth=2, color='red')
    plt.semilogy (lstm_gradients, label='LSTM', linewidth=2, color='blue')
    
    plt.xlabel('Timesteps Back', fontsize=12)
    plt.ylabel('Gradient Magnitude (log scale)', fontsize=12)
    plt.title('Gradient Flow: RNN vs LSTM', fontsize=14)
    plt.legend (fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Gradient Magnitudes:")
    print("=" * 60)
    for t in [10, 25, 50, 100]:
        rnn_grad = rnn_gradients[t-1] if t <= len (rnn_gradients) else 0
        lstm_grad = lstm_gradients[t-1] if t <= len (lstm_gradients) else 0
        print(f"After {t} steps:")
        print(f"  RNN:  {rnn_grad:.2e}")
        print(f"  LSTM: {lstm_grad:.4f}")
        print()
    
    print("Key Insight:")
    print("→ RNN gradients vanish exponentially")
    print("→ LSTM gradients decay slowly (controlled by forget gate)")
    print("→ LSTM can learn dependencies 100+ steps away!")

compare_gradient_flow()
\`\`\`

## GRU (Gated Recurrent Unit)

### Simplified LSTM

GRU was proposed as a simpler alternative to LSTM with fewer parameters but similar performance.

**Key Differences from LSTM**:
1. **No separate cell state** - only hidden state
2. **Two gates instead of three**:
   - Update gate (combines forget and input gates)
   - Reset gate (controls use of previous hidden state)
3. **Fewer parameters** (~25% less than LSTM)

**GRU Equations**:
\`\`\`
Reset Gate:  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
Update Gate: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
Candidate:   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
Hidden:      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
\`\`\`

\`\`\`python
# GRU in PyTorch
class CharacterGRU(nn.Module):
    """
    Character-level language model with GRU.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(CharacterGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding (vocab_size, embedding_dim)
        
        # GRU layer (s)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear (hidden_size, vocab_size)
    
    def forward (self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_length)
            hidden: (num_layers, batch, hidden_size) or None
        
        Returns:
            output: (batch, seq_length, vocab_size)
            hidden: (num_layers, batch, hidden_size)
        """
        embedded = self.embedding (x)
        gru_out, hidden = self.gru (embedded, hidden)
        
        # Reshape and project
        batch_size, seq_length, _ = gru_out.shape
        gru_out = gru_out.contiguous().view(-1, self.hidden_size)
        output = self.fc (gru_out)
        output = output.view (batch_size, seq_length, -1)
        
        return output, hidden

# Compare LSTM vs GRU parameters
lstm_model = CharacterLSTM(vocab_size, embedding_dim, hidden_size, num_layers)
gru_model = CharacterGRU(vocab_size, embedding_dim, hidden_size, num_layers)

lstm_params = count_params (lstm_model)
gru_params = count_params (gru_model)

print("LSTM vs GRU Comparison:")
print("=" * 60)
print(f"LSTM parameters: {lstm_params:,}")
print(f"GRU parameters:  {gru_params:,}")
print(f"Difference: {lstm_params - gru_params:,} ({(1 - gru_params/lstm_params)*100:.1f}% fewer in GRU)")

# Test both
x = torch.randint(0, vocab_size, (batch_size, seq_length))

lstm_out, _ = lstm_model (x)
gru_out, _ = gru_model (x)

print(f"\\nOutput shapes (both identical):")
print(f"  LSTM: {lstm_out.shape}")
print(f"  GRU:  {gru_out.shape}")
\`\`\`

## LSTM vs GRU: When to Use Which?

\`\`\`python
import pandas as pd

comparison = pd.DataFrame([
    {
        'Aspect': 'Parameters',
        'LSTM': 'More (~33% more)',
        'GRU': 'Fewer',
        'Winner': 'GRU'
    },
    {
        'Aspect': 'Training Speed',
        'LSTM': 'Slower',
        'GRU': 'Faster (~25%)',
        'Winner': 'GRU'
    },
    {
        'Aspect': 'Memory Usage',
        'LSTM': 'Higher (2 states)',
        'GRU': 'Lower (1 state)',
        'Winner': 'GRU'
    },
    {
        'Aspect': 'Expressiveness',
        'LSTM': 'More flexible',
        'GRU': 'Simpler',
        'Winner': 'LSTM'
    },
    {
        'Aspect': 'Long Sequences',
        'LSTM': 'Slight edge',
        'GRU': 'Good',
        'Winner': 'LSTM'
    },
    {
        'Aspect': 'Small Data',
        'LSTM': 'Can overfit',
        'GRU': 'Better generalization',
        'Winner': 'GRU'
    },
])

print("\\nLSTM vs GRU Detailed Comparison:")
print("=" * 80)
print(comparison.to_string (index=False))

print("\\n\\nWhen to Use LSTM:")
print("✓ Large datasets (millions of samples)")
print("✓ Very long sequences (>1000 timesteps)")
print("✓ Complex patterns requiring more parameters")
print("✓ When memory/speed not critical")
print("✓ Bidirectional models (LSTM slightly better)")

print("\\nWhen to Use GRU:")
print("✓ Limited computational resources")
print("✓ Faster iteration during experimentation")
print("✓ Small to medium datasets")
print("✓ Moderate sequence lengths (<500 timesteps)")
print("✓ Default choice for many practitioners")

print("\\nRule of Thumb:")
print("→ Start with GRU (faster, simpler)")
print("→ Switch to LSTM if you need the extra capacity")
print("→ In practice, performance often similar")
print("→ Modern trend: Use Transformers for both! (when possible)")
\`\`\`

## Practical Training Tips

\`\`\`python
# Best practices for training LSTM/GRU models

def train_lstm_best_practices():
    """Demonstrate LSTM/GRU training best practices."""
    
    print("LSTM/GRU Training Best Practices:")
    print("=" * 60)
    
    print("\\n1. GRADIENT CLIPPING (Essential!)")
    print("   - Prevents exploding gradients")
    print("   - Use torch.nn.utils.clip_grad_norm_()")
    print("   - Typical threshold: 1-5")
    print("   \`\`\`python")
    print("   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)")
    print("   \`\`\`")
    
    print("\\n2. DROPOUT")
    print("   - Apply between LSTM layers (if multi-layer)")
    print("   - Typical: 0.2-0.5")
    print("   - Built into PyTorch LSTM:")
    print("   \`\`\`python")
    print("   nn.LSTM(..., dropout=0.3, num_layers=2)")
    print("   \`\`\`")
    
    print("\\n3. LEARNING RATE")
    print("   - Start lower than CNNs: 0.001-0.0001")
    print("   - Use learning rate scheduling")
    print("   - Warmup can help")
    
    print("\\n4. BATCH SIZE")
    print("   - Smaller than CNNs (memory intensive)")
    print("   - Typical: 32-128")
    print("   - Trade-off: speed vs stability")
    
    print("\\n5. SEQUENCE LENGTH")
    print("   - Start with shorter sequences (50-100)")
    print("   - Gradually increase if needed")
    print("   - Use truncated BPTT for very long sequences")
    
    print("\\n6. INITIALIZATION")
    print("   - Xavier/Glorot for weights")
    print("   - Forget gate bias: Initialize to 1")
    print("   - Helps remember information initially")
    
    print("\\n7. LAYER NORMALIZATION")
    print("   - Apply to LSTM hidden states")
    print("   - Stabilizes training")
    print("   - Alternative to Batch Norm for sequences")
    
    print("\\n8. BIDIRECTIONAL")
    print("   - For non-causal tasks (classification)")
    print("   - Processes sequence both directions")
    print("   - Doubles parameters and computation")

train_lstm_best_practices()
\`\`\`

## Key Takeaways

1. **LSTM solves vanishing gradients** through gating mechanisms and cell state

2. **Three gates in LSTM**:
   - Forget gate: What to remove
   - Input gate: What to add
   - Output gate: What to expose

3. **Cell state** acts as a "memory highway" allowing gradients to flow unchanged

4. **GRU** is a simpler alternative:
   - Fewer parameters (~25% less)
   - Faster training
   - Similar performance

5. **LSTM vs GRU**:
   - Start with GRU (faster experimentation)
   - Use LSTM for very long sequences or large datasets
   - Performance often similar in practice

6. **Training tips**:
   - Gradient clipping essential
   - Lower learning rates than CNNs
   - Dropout between layers
   - Layer normalization helps

7. **Modern context**: Transformers have largely replaced LSTM/GRU for many NLP tasks, but LSTM/GRU still excel for:
   - Time series forecasting
   - Streaming data
   - Resource-constrained environments

## Coming Next

In the next section, we'll explore **Sequence-to-Sequence Models** - using encoder-decoder architectures for tasks like machine translation!
`,
};
