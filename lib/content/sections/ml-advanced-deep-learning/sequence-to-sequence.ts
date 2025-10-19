/**
 * Sequence-to-Sequence Models Section
 */

export const sequenceToSequenceSection = {
  id: 'sequence-to-sequence',
  title: 'Sequence-to-Sequence Models',
  content: `# Sequence-to-Sequence Models

## Introduction

Sequence-to-Sequence (Seq2Seq) models are a powerful architecture for tasks where both input and output are sequences of variable length. They form the foundation for machine translation, text summarization, chatbots, and more.

**Key Applications**:
- **Machine Translation**: English → French
- **Text Summarization**: Article → summary
- **Question Answering**: Context + question → answer
- **Code Generation**: Description → code
- **Speech Recognition**: Audio → text

**Key Challenge**: Input and output sequences have **different lengths** and are not aligned.

## The Encoder-Decoder Architecture

### Core Idea

Seq2Seq consists of two main components:

1. **Encoder**: Reads input sequence, compresses into fixed-size context vector
2. **Decoder**: Generates output sequence from context vector

\`\`\`
Input Sequence → [ENCODER] → Context Vector → [DECODER] → Output Sequence
   x_1,...,x_n       RNN/LSTM        c (fixed)      RNN/LSTM      y_1,...,y_m
\`\`\`

**Key Insight**: Context vector = "thought" that captures meaning of entire input sequence

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Encoder: Input sequence → Context vector
    """
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers,
            batch_first=True
        )
    
    def forward(self, x):
        """
        Encode input sequence.
        
        Args:
            x: (batch, seq_length) - input token indices
        
        Returns:
            outputs: (batch, seq_length, hidden_size)
            hidden: tuple (h_n, c_n) - final states (context)
        """
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq, embedding_dim)
        
        # Encode with LSTM
        outputs, hidden = self.lstm(embedded)
        # outputs: (batch, seq, hidden_size)
        # hidden: ((num_layers, batch, hidden_size), (num_layers, batch, hidden_size))
        
        return outputs, hidden

class Decoder(nn.Module):
    """
    Decoder: Context vector → Output sequence
    """
    def __init__(self, output_size, embedding_dim, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        """
        Decode one step.
        
        Args:
            x: (batch, 1) - current input token
            hidden: Previous hidden state (context)
        
        Returns:
            output: (batch, output_size) - predicted token logits
            hidden: Updated hidden state
        """
        # Embed token
        embedded = self.embedding(x)  # (batch, 1, embedding_dim)
        
        # Decode one step
        output, hidden = self.lstm(embedded, hidden)
        # output: (batch, 1, hidden_size)
        
        # Project to vocabulary
        output = self.fc(output.squeeze(1))  # (batch, output_size)
        
        return output, hidden

class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq model.
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """
        Forward pass through Seq2Seq.
        
        Args:
            source: (batch, source_length) - input sequence
            target: (batch, target_length) - target sequence
            teacher_forcing_ratio: Probability of using true target as next input
        
        Returns:
            outputs: (batch, target_length, output_size) - predictions
        """
        batch_size = source.size(0)
        target_length = target.size(1)
        output_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_length, output_size).to(self.device)
        
        # Encode entire input sequence
        _, hidden = self.encoder(source)
        
        # First input to decoder is <START> token
        decoder_input = target[:, 0].unsqueeze(1)  # (batch, 1)
        
        # Decode one token at a time
        for t in range(1, target_length):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output
            outputs[:, t, :] = output
            
            # Teacher forcing: use true target as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1)  # (batch,)
            
            # Next input: true target or prediction
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs

# Create Seq2Seq model
print("Seq2Seq Model Architecture:")
print("=" * 60)

input_vocab_size = 10000   # Source language vocabulary
output_vocab_size = 8000   # Target language vocabulary
embedding_dim = 256
hidden_size = 512
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(input_vocab_size, embedding_dim, hidden_size, num_layers)
decoder = Decoder(output_vocab_size, embedding_dim, hidden_size, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

print(f"Encoder: {input_vocab_size} vocab → {hidden_size} hidden")
print(f"Decoder: {hidden_size} hidden → {output_vocab_size} vocab")

# Count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\\nTotal parameters: {count_params(model):,}")
print(f"  Encoder: {count_params(encoder):,}")
print(f"  Decoder: {count_params(decoder):,}")

# Test forward pass
batch_size = 32
source_length = 20
target_length = 25

source = torch.randint(0, input_vocab_size, (batch_size, source_length)).to(device)
target = torch.randint(0, output_vocab_size, (batch_size, target_length)).to(device)

outputs = model(source, target, teacher_forcing_ratio=0.5)

print(f"\\nExample Forward Pass:")
print(f"  Source shape: {source.shape}")
print(f"  Target shape: {target.shape}")
print(f"  Output shape: {outputs.shape}")
print(f"\\n→ Input: 20 tokens")
print(f"→ Output: 25 tokens (different length!)")
\`\`\`

## The Bottleneck Problem

### Context Vector Limitation

**Problem**: All information from input sequence must be compressed into fixed-size context vector (typically 512-1024 dimensions).

**Issues**:
1. **Information Loss**: Long sequences lose information in compression
2. **Fixed Size**: Same size context for 5-word vs 50-word input
3. **Distant Dependencies**: Hard to remember early tokens when generating late outputs

**Example**:
\`\`\`
Input: "The quick brown fox jumps over the lazy dog in the park"
                                                    ↓
Context: [512-dim vector] ← Must capture ALL this information!
                                                    ↓
Output: "Le rapide renard brun saute par-dessus le chien paresseux dans le parc"
\`\`\`

\`\`\`python
# Visualize context vector bottleneck
def visualize_bottleneck():
    """Show how information flows through bottleneck."""
    
    import matplotlib.pyplot as plt
    
    # Simulate sequence lengths
    input_lengths = np.array([5, 10, 20, 50, 100])
    
    # Context vector size (fixed)
    context_size = 512
    
    # Information loss (hypothetical)
    # Longer sequences lose more information
    information_retained = 100 * np.exp(-input_lengths / 30)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bottleneck visualization
    ax1.barh([1], [input_lengths[0]], color='blue', alpha=0.3, label='Input')
    ax1.barh([2], [context_size], color='red', label='Context (bottleneck)')
    ax1.barh([3], [input_lengths[0]], color='green', alpha=0.3, label='Output')
    
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Input\\n(variable)', 'Context\\n(fixed)', 'Output\\n(variable)'])
    ax1.set_xlabel('Dimensions/Tokens')
    ax1.set_title('The Bottleneck Problem')
    ax1.legend()
    
    # Plot 2: Information retention
    ax2.plot(input_lengths, information_retained, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Input Sequence Length')
    ax2.set_ylabel('Information Retained (%)')
    ax2.set_title('Information Loss in Long Sequences')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='r', linestyle='--', label='50% threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Bottleneck Effects:")
    print("=" * 60)
    for length, retained in zip(input_lengths, information_retained):
        print(f"  {length:3d} tokens → {retained:5.1f}% information retained")
    
    print("\\n→ Fixed-size context vector can't scale to long sequences!")
    print("→ Solution: Attention mechanism (next section)!")

visualize_bottleneck()
\`\`\`

## Teacher Forcing

### Training Strategy

**Teacher Forcing**: During training, feed the **true previous output** (not model's prediction) as next input to decoder.

**Without Teacher Forcing** (free running):
\`\`\`
Step 1: Input <START> → Predict "Le"
Step 2: Input "Le" (predicted) → Predict "chat"  
Step 3: Input "chat" (predicted) → Predict "mange"
...
\`\`\`
Problem: If step 1 wrong, error compounds!

**With Teacher Forcing**:
\`\`\`
Step 1: Input <START> → Predict "Le"
Step 2: Input "Le" (TRUE) → Predict "chat"
Step 3: Input "chat" (TRUE) → Predict "mange"
...
\`\`\`
Benefit: Each step learns from correct context

\`\`\`python
# Training with teacher forcing
def train_seq2seq(model, iterator, optimizer, criterion, clip, device):
    """
    Train Seq2Seq model for one epoch.
    
    Args:
        model: Seq2Seq model
        iterator: Data iterator
        optimizer: Optimizer
        criterion: Loss function
        clip: Gradient clipping value
        device: Device
    """
    model.train()
    
    epoch_loss = 0
    
    for batch in iterator:
        source = batch.source.to(device)
        target = batch.target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        output = model(source, target, teacher_forcing_ratio=0.5)
        
        # output: (batch, target_length, output_vocab)
        # target: (batch, target_length)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        
        # Ignore first token (<START>)
        output = output[:, 1:].contiguous().view(-1, output_dim)
        target = target[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Inference (no teacher forcing)
def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_length=50):
    """
    Translate a sentence using trained model.
    
    Args:
        model: Trained Seq2Seq model
        sentence: Input sentence (string)
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        device: Device
        max_length: Maximum output length
    """
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    indices = [src_vocab.stoi[token] for token in tokens]
    indices = [src_vocab.stoi['<sos>']] + indices + [src_vocab.stoi['<eos>']]
    
    source_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)  # (1, seq)
    
    with torch.no_grad():
        # Encode
        _, hidden = model.encoder(source_tensor)
        
        # Start decoding
        decoder_input = torch.LongTensor([trg_vocab.stoi['<sos>']]).to(device)
        
        outputs = []
        
        for _ in range(max_length):
            # Decode one step
            output, hidden = model.decoder(decoder_input.unsqueeze(0), hidden)
            
            # Get predicted token
            predicted_token = output.argmax(1).item()
            
            # Stop if <EOS>
            if predicted_token == trg_vocab.stoi['<eos>']:
                break
            
            outputs.append(predicted_token)
            
            # Next input is predicted token (no teacher forcing)
            decoder_input = torch.LongTensor([predicted_token]).to(device)
    
    # Convert indices to tokens
    translated_tokens = [trg_vocab.itos[idx] for idx in outputs]
    
    return ' '.join(translated_tokens)

print("\\nTeacher Forcing:")
print("=" * 60)
print("Training (ratio=0.5):")
print("  → 50% of time: use true target")
print("  → 50% of time: use model prediction")
print("  → Balances fast learning with exposure to errors")
print("\\nInference:")
print("  → Always use model's own predictions")
print("  → No access to true targets!")
print("\\nExposure Bias:")
print("  → Model never sees its own errors during training")
print("  → Can struggle when predictions accumulate errors")
print("  → Solution: Scheduled sampling (gradually reduce teacher forcing)")
\`\`\`

## Handling Variable-Length Sequences

### Padding and Masking

Real-world batches have sequences of different lengths. Solution: **padding** + **masking**.

\`\`\`python
# Padding and masking for variable-length sequences

def pad_sequences(sequences, max_length, pad_token=0):
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of sequences (variable length)
        max_length: Maximum sequence length
        pad_token: Token to use for padding
    
    Returns:
        padded: (batch, max_length) tensor
        lengths: List of original lengths
    """
    padded = []
    lengths = []
    
    for seq in sequences:
        length = len(seq)
        lengths.append(length)
        
        if length < max_length:
            # Pad
            seq = seq + [pad_token] * (max_length - length)
        else:
            # Truncate
            seq = seq[:max_length]
        
        padded.append(seq)
    
    return torch.LongTensor(padded), lengths

# Example
sequences = [
    [1, 2, 3, 4, 5],           # Length 5
    [1, 2, 3],                 # Length 3
    [1, 2, 3, 4, 5, 6, 7, 8],  # Length 8
]

padded, lengths = pad_sequences(sequences, max_length=10, pad_token=0)

print("Variable-Length Sequences:")
print("=" * 60)
print(f"Original sequences: {sequences}")
print(f"\\nPadded sequences:")
print(padded)
print(f"\\nOriginal lengths: {lengths}")
print(f"\\nPadded tensor shape: {padded.shape}")

# Create attention mask
def create_padding_mask(lengths, max_length):
    """
    Create mask for padded positions.
    
    Returns:
        mask: (batch, max_length) with 1 for real tokens, 0 for padding
    """
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    return mask

mask = create_padding_mask(lengths, max_length=10)
print(f"\\nPadding mask:")
print(mask)
print("\\n→ 1 = real token, 0 = padding")
print("→ Loss computed only on real tokens (ignore padding)")

# Masked loss calculation
def masked_cross_entropy_loss(predictions, targets, mask):
    """
    Cross-entropy loss ignoring padded positions.
    
    Args:
        predictions: (batch, seq, vocab_size)
        targets: (batch, seq)
        mask: (batch, seq) - 1 for real, 0 for pad
    """
    # Flatten
    predictions = predictions.view(-1, predictions.size(-1))
    targets = targets.view(-1)
    mask = mask.view(-1)
    
    # Compute loss
    loss = F.cross_entropy(predictions, targets, reduction='none')
    
    # Apply mask
    masked_loss = loss * mask
    
    # Average over non-padded tokens
    return masked_loss.sum() / mask.sum()

print("\\nMasked Loss:")
print("→ Computes loss only on real tokens")
print("→ Ignores padded positions")
print("→ Fair comparison across different lengths")
\`\`\`

## Bidirectional Encoder

### Looking Both Ways

For encoder (non-autoregressive), can process sequence in **both directions**:

**Forward LSTM**: Left to right
**Backward LSTM**: Right to left
**Concatenate**: Richer representation

\`\`\`python
class BidirectionalEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.
    """
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=1):
        super(BidirectionalEncoder, self).__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True  # Key parameter!
        )
        
        # Project concatenated hidden states to decoder hidden size
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        """
        Bidirectional encoding.
        
        Returns:
            outputs: (batch, seq, hidden_size * 2) - forward + backward
            hidden: Processed to match decoder size
        """
        embedded = self.embedding(x)
        
        # Bidirectional LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, seq, hidden_size * 2)
        # hidden: (num_layers * 2, batch, hidden_size)
        
        # Combine forward and backward hidden states
        # hidden[-2]: last forward hidden state
        # hidden[-1]: last backward hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        # hidden: (batch, hidden_size)
        
        # Reshape for decoder
        hidden = hidden.unsqueeze(0)  # (1, batch, hidden_size)
        
        return outputs, (hidden, hidden)  # Use same for h and c

# Compare unidirectional vs bidirectional
print("\\nBidirectional Encoder:")
print("=" * 60)
print("Unidirectional:")
print("  → Processes left-to-right only")
print("  → Hidden state at position i has info from [0...i]")
print("\\nBidirectional:")
print("  → Forward pass: left-to-right")
print("  → Backward pass: right-to-left")
print("  → Concatenate: [forward_hidden, backward_hidden]")
print("  → Hidden state at position i has info from ENTIRE sequence")
print("\\nAdvantages:")
print("  ✓ Better context understanding")
print("  ✓ Richer representations")
print("  ✓ Improved translation quality")
print("\\nDisadvantages:")
print("  ✗ 2× parameters and computation")
print("  ✗ Not suitable for decoder (autoregressive)")
\`\`\`

## Beam Search

### Better Decoding Strategy

**Greedy Decoding**: Choose highest probability token at each step
- Fast but myopic
- Can lead to suboptimal sequences

**Beam Search**: Maintain top-k hypotheses at each step
- Explores multiple paths
- Better quality outputs

\`\`\`python
def beam_search_decode(model, source, beam_width=5, max_length=50):
    """
    Beam search decoding.
    
    Args:
        model: Trained Seq2Seq model
        source: Source sequence
        beam_width: Number of beams to maintain
        max_length: Maximum output length
    
    Returns:
        Best sequence and its score
    """
    model.eval()
    
    with torch.no_grad():
        # Encode
        _, hidden = model.encoder(source)
        
        # Initialize beams
        # Each beam: (sequence, score, hidden_state)
        beams = [([trg_vocab.stoi['<sos>']], 0.0, hidden)]
        
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            # Expand each beam
            for seq, score, hidden in beams:
                # Stop if EOS
                if seq[-1] == trg_vocab.stoi['<eos>']:
                    completed.append((seq, score))
                    continue
                
                # Decode one step
                decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0)
                output, new_hidden = model.decoder(decoder_input, hidden)
                
                # Get top beam_width predictions
                log_probs = F.log_softmax(output, dim=-1)
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
                # Create new candidates
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score, new_hidden))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Stop if all beams completed
            if not beams:
                break
        
        # Add remaining beams to completed
        completed.extend(beams)
        
        # Return best sequence
        completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)  # Normalize by length
        best_seq, best_score = completed[0][:2]
        
        return best_seq, best_score

print("\\nBeam Search:")
print("=" * 60)
print("Beam Width = 1 (Greedy):")
print("  Step 1: 'The' (p=0.8)")
print("  Step 2: 'cat' (p=0.6)")
print("  Final: p = 0.8 × 0.6 = 0.48")
print("\\nBeam Width = 3:")
print("  Step 1: Keep top 3")
print("    1. 'The' (p=0.8)")
print("    2. 'A' (p=0.1)")
print("    3. 'This' (p=0.05)")
print("  Step 2: Expand each, keep top 3 overall")
print("    1. 'The', 'cat' (p=0.48)")
print("    2. 'The', 'dog' (p=0.24)")
print("    3. 'A', 'cat' (p=0.09)")
print("  ... continues ...")
print("\\nAdvantages:")
print("  ✓ Better quality than greedy")
print("  ✓ Explores alternative paths")
print("  ✓ Finds globally better sequences")
print("\\nTrade-off:")
print("  → beam_width=1: Fast, lower quality")
print("  → beam_width=10: Slower, higher quality")
print("  → beam_width=50: Much slower, marginal improvement")
print("  → Typical: beam_width=5-10")
\`\`\`

## Key Takeaways

1. **Seq2Seq** = Encoder-Decoder architecture for variable-length sequence transduction

2. **Encoder** compresses input into fixed-size context vector

3. **Decoder** generates output autoregressively from context

4. **Bottleneck problem**: Fixed-size context limits long sequences (solved by attention!)

5. **Teacher forcing**: Use true targets during training for faster convergence

6. **Padding & masking**: Handle variable-length sequences in batches

7. **Bidirectional encoder**: Process both directions for better representations

8. **Beam search**: Better decoding than greedy, explores multiple hypotheses

9. **Limitation**: Context vector bottleneck prevents excellent performance on long sequences

## Coming Next

In the next section, we'll explore the **Attention Mechanism** - the breakthrough that solves the bottleneck problem and revolutionized sequence modeling!
`,
};
