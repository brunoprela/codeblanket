/**
 * Section: Sequence Modeling for NLP
 * Module: Natural Language Processing
 *
 * Covers RNNs, LSTMs, GRUs, and sequence modeling for text processing
 */

export const sequenceModelingNlpSection = {
  id: 'sequence-modeling-nlp',
  title: 'Sequence Modeling for NLP',
  content: `
# Sequence Modeling for NLP

## Introduction

Text is inherently sequential - word order matters. "Dog bites man" is very different from "Man bites dog." Sequence models process text while maintaining this temporal structure, making them ideal for NLP tasks.

**Why Sequence Models:**
- **Order matters**: Capture word dependencies
- **Variable length**: Handle sentences of any length
- **Context propagation**: Information flows through sequence
- **Recurrence**: Process one element at a time, maintaining state

This section covers RNNs, LSTMs, and GRUs - the foundation for understanding modern NLP architectures.

## Recurrent Neural Networks (RNNs)

### Basic RNN Architecture

\`\`\`python
import numpy as np

class SimpleRNN:
    """Basic RNN from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
    
    def forward(self, inputs):
        """
        inputs: list of input vectors (sequence)
        Returns: outputs and hidden states for each timestep
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        outputs = []
        hidden_states = []
        
        for x in inputs:
            # RNN cell computation
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            hidden_states.append(h)
            outputs.append(y)
        
        return outputs, hidden_states

# Example usage
vocab_size = 10
hidden_size = 20
output_size = 10

rnn = SimpleRNN(vocab_size, hidden_size, output_size)

# Sequence: "I love NLP"
sequence = [
    np.random.randn(vocab_size, 1),  # "I"
    np.random.randn(vocab_size, 1),  # "love"
    np.random.randn(vocab_size, 1),  # "NLP"
]

outputs, hidden_states = rnn.forward(sequence)
print(f"Number of outputs: {len(outputs)}")
print(f"Output shape: {outputs[0].shape}")
\`\`\`

### RNN with PyTorch

\`\`\`python
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        output, hidden = self.rnn(embedded)  # output: (batch, seq_len, hidden_dim)
        
        # Use final hidden state for classification
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        logits = self.fc(final_hidden)  # (batch, output_dim)
        
        return logits

# Initialize model
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary classification

model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Example input
batch_size = 32
seq_len = 20
x = torch.randint(0, vocab_size, (batch_size, seq_len))

logits = model(x)
print(f"Output shape: {logits.shape}")  # (32, 2)
\`\`\`

## The Vanishing Gradient Problem

\`\`\`python
# Demonstration of vanishing gradients

# RNN unrolled through time:
# h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)

# During backpropagation:
# gradient flows backward through time
# Multiplied by tanh derivative at each step

# tanh derivative ≤ 0.25
# After many timesteps: 0.25^n → very small!

# Example with 50 timesteps:
import numpy as np
gradient_scale = 0.25 ** 50
print(f"Gradient after 50 steps: {gradient_scale:.2e}")  # ~10^-30 (vanishes!)

# This makes learning long-term dependencies nearly impossible
\`\`\`

## Long Short-Term Memory (LSTM)

LSTMs solve the vanishing gradient problem with gates that control information flow.

### LSTM Architecture

\`\`\`python
class LSTMCell:
    """LSTM cell from scratch"""
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Gates: forget, input, output
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, h_prev, c_prev):
        """
        x: input at current timestep
        h_prev: previous hidden state
        c_prev: previous cell state
        """
        # Concatenate input and previous hidden state
        combined = np.vstack((x, h_prev))
        
        # Forget gate: what to forget from cell state
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        
        # Input gate: what new information to add
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        
        # Candidate cell state
        c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        
        # Update cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate: what to output
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Hidden state
        h = o * np.tanh(c)
        
        return h, c

# Intuition:
# - Forget gate: "Should I forget this information?"
# - Input gate: "Should I update with new information?"
# - Output gate: "What should I output?"
# - Cell state: Long-term memory (gradients flow easily!)
\`\`\`

### LSTM with PyTorch

\`\`\`python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            batch_first=True,
            dropout=0.5  # Dropout between layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM returns output and (hidden, cell) states
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden state
        final_hidden = hidden[-1]  # Last layer
        final_hidden = self.dropout(final_hidden)
        
        logits = self.fc(final_hidden)
        return logits

# Training example
model = LSTMModel(vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        texts, labels = batch
        
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
\`\`\`

## Gated Recurrent Unit (GRU)

GRUs simplify LSTMs with fewer gates, often similar performance.

\`\`\`python
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        logits = self.fc(hidden[-1])
        return logits

# GRU has 2 gates vs LSTM's 3 gates:
# - Reset gate: controls how much past information to forget
# - Update gate: controls how much new information to add
# No separate cell state (simpler than LSTM)
\`\`\`

## Bidirectional RNNs

\`\`\`python
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            bidirectional=True,  # Key parameter
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.bilstm(embedded)
        
        # Concatenate forward and backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        logits = self.fc(combined)
        return logits

# Bidirectional processes sequence in both directions
# Better for understanding but cannot be used for generation
\`\`\`

## Sequence-to-Sequence Models

\`\`\`python
class Seq2SeqModel(nn.Module):
    """Encoder-Decoder for sequence-to-sequence tasks"""
    
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, output_vocab_size)
        
    def forward(self, source, target):
        # Encode
        embedded_source = self.encoder_embedding(source)
        encoder_output, (hidden, cell) = self.encoder_lstm(embedded_source)
        
        # Decode (using encoder's final state as initial state)
        embedded_target = self.decoder_embedding(target)
        decoder_output, _ = self.decoder_lstm(embedded_target, (hidden, cell))
        
        # Predict output vocabulary
        logits = self.decoder_output(decoder_output)
        return logits

# Applications: Machine translation, summarization
\`\`\`

## Text Generation with LSTMs

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def generate(self, start_token, max_length=50, temperature=1.0):
        """Generate text autoregressively"""
        self.eval()
        generated = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.LongTensor([[generated[-1]]])
                logits, hidden = self.forward(x, hidden)
                
                # Sample from distribution (with temperature)
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                if next_token == END_TOKEN:
                    break
        
        return generated

# Temperature controls randomness:
# - temperature < 1.0: More deterministic (peaks)
# - temperature = 1.0: Sample from actual distribution
# - temperature > 1.0: More random (flattens)
\`\`\`

## Sentiment Analysis with LSTM

\`\`\`python
# Complete sentiment analysis example
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        return self.fc(hidden)

# Training
model = SentimentLSTM(vocab_size=10000)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(10):
    for batch in train_loader:
        texts, labels = batch
        
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        
        # Gradient clipping (important for RNNs!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
\`\`\`

## Best Practices

1. **Gradient Clipping**: Essential for RNNs to prevent exploding gradients
2. **Bidirectional for understanding**: Use BiLSTM when you can see full sequence
3. **Layer normalization**: Stabilizes training
4. **Dropout**: Between layers and embeddings
5. **Embedding dimension**: 100-300 typically sufficient
6. **Hidden dimension**: 256-512 for most tasks
7. **Number of layers**: 2-3 layers usually best (more overfits)

## When to Use Sequence Models

**Use LSTMs/GRUs when:**
- Sequential processing is natural
- Don't need full parallelization
- Generating sequences (can't use bidirectional transformers)
- Memory constraints (smaller than transformers)

**Use Transformers when:**
- Need best accuracy
- Can parallelize training
- Have sufficient compute
- Full bidirectional context needed

**Key Insight:**
While transformers dominate modern NLP, understanding LSTMs remains valuable for:
- Generating sequences
- Low-resource settings
- Understanding sequence modeling fundamentals
- Hybrid architectures
`,
};
