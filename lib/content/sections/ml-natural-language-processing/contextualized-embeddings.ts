/**
 * Section: Contextualized Embeddings
 * Module: Natural Language Processing
 *
 * Covers ELMo and the transition from static to context-dependent embeddings
 */

export const contextualizedEmbeddingsSection = {
  id: 'contextualized-embeddings',
  title: 'Contextualized Embeddings',
  content: `
# Contextualized Embeddings

## Introduction

Contextualized embeddings revolutionized NLP by solving the polysemy problem: instead of assigning one fixed vector per word, they generate different embeddings for the same word based on context. The word "bank" receives different vectors in "financial bank" vs "river bank."

**Key Innovation:**
- **Context-dependent**: Same word, different contexts → different embeddings
- **Dynamic representations**: Computed on-the-fly for each occurrence
- **Solves polysemy**: Disambiguates word meanings automatically
- **Transfer learning**: Pre-trained models capture linguistic knowledge

**Historical Impact:**
ELMo (2018) was the breakthrough that sparked the transformer revolution (BERT, GPT), fundamentally changing how we do NLP.

## The Polysemy Problem Revisited

Static embeddings (Word2Vec, GloVe) fail on polysemous words:

\`\`\`python
# Static embedding example
word2vec_model.wv['bank']  
# Always returns same vector, whether:
# - "I deposited money at the bank" (financial)
# - "We sat on the river bank" (geographical)

# The problem: ONE vector must represent ALL meanings
# Result: Compromised representation
\`\`\`

**Contextualized Solution:**
\`\`\`python
# Contextualized embedding (conceptual)
embed (sentence="I deposited money at the bank", word_position=5)
# Returns vector specific to financial context

embed (sentence="We sat on the river bank", word_position=5)
# Returns different vector specific to geographical context
\`\`\`

## ELMo: Embeddings from Language Models

**ELMo (Embeddings from Language Models)** by Peters et al. (2018) uses bidirectional LSTMs to generate context-dependent representations.

### ELMo Architecture

\`\`\`
Architecture Overview:

Input: Character-level tokens
    ↓
Character CNN (handles OOV)
    ↓
Bidirectional LSTM Layer 1 →→→→ (forward)
                            ←←←← (backward)
    ↓
Bidirectional LSTM Layer 2 →→→→
                            ←←←←
    ↓
Weighted combination of layers
    ↓
Contextualized embedding for each word
\`\`\`

**Key Features:**
1. **Bidirectional**: Reads text forward and backward
2. **Deep**: Multiple LSTM layers capture different linguistic aspects
3. **Character-based**: Handles OOV words via character CNN
4. **Pre-trained**: Trained on large corpus (1 Billion Word Benchmark)

### How ELMo Works

\`\`\`python
# Conceptual ELMo operation

sentence = "I love machine learning"

# For word "machine" at position 2:
# 1. Forward LSTM reads: "I love machine" → hidden state
# 2. Backward LSTM reads: "learning machine love I" → hidden state
# 3. Combine both directions at each layer
# 4. Weight layers: embedding = w₀·layer₀ + w₁·layer₁ + w₂·layer₂

# Different sentences → different contexts → different embeddings
sentence1 = "I love machine learning"
sentence2 = "The machine broke down"
# embed("machine", sentence1) ≠ embed("machine", sentence2)
\`\`\`

### Using Pre-trained ELMo

\`\`\`python
# Install: pip install allennlp

from allennlp.commands.elmo import ElmoEmbedder

# Load pre-trained ELMo
elmo = ElmoEmbedder(
    options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
    weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
    cuda_device=-1  # -1 for CPU, 0+ for GPU
)

# Embed sentences
sentences = [
    ['I', 'deposited', 'money', 'at', 'the', 'bank'],
    ['We', 'sat', 'on', 'the', 'river', 'bank']
]

# Get embeddings
embeddings = elmo.embed_sentences (sentences)

# embeddings[0] = sentence 1, shape: (3, 6, 1024)
# - 3 layers (0=character, 1=LSTM1, 2=LSTM2)
# - 6 words
# - 1024 dimensions per word

# Get embedding for "bank" in first sentence
bank_financial = embeddings[0][:, 5, :]  # Position 5, all layers
print(f"Bank (financial) shape: {bank_financial.shape}")  # (3, 1024)

# Get embedding for "bank" in second sentence  
bank_river = embeddings[1][:, 5, :]  # Position 5, all layers
print(f"Bank (river) shape: {bank_river.shape}")  # (3, 1024)

# Compare embeddings
from scipy.spatial.distance import cosine

# Average across layers for comparison
bank_financial_avg = bank_financial.mean (axis=0)
bank_river_avg = bank_river.mean (axis=0)

similarity = 1 - cosine (bank_financial_avg, bank_river_avg)
print(f"Similarity between 'bank' in different contexts: {similarity:.4f}")
# Lower similarity than static embeddings!
\`\`\`

### ELMo for Downstream Tasks

\`\`\`python
import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

# Initialize ELMo
options_file = "elmo_options.json"
weight_file = "elmo_weights.hdf5"

elmo = Elmo (options_file, weight_file, num_output_representations=1, dropout=0)

# Prepare sentences
sentences = [
    ['The', 'cat', 'sat', 'on', 'the', 'mat'],
    ['Deep', 'learning', 'is', 'amazing']
]

# Convert to character IDs
character_ids = batch_to_ids (sentences)

# Get ELMo embeddings
embeddings = elmo (character_ids)

# embeddings['elmo_representations'][0] shape: (2, max_len, 1024)
# - 2 sentences
# - Variable length (padded to max)
# - 1024-dim embeddings

# Use embeddings as features for classifier
class TextClassifier (nn.Module):
    def __init__(self, elmo_model, num_classes):
        super().__init__()
        self.elmo = elmo_model
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward (self, character_ids):
        # Get ELMo embeddings
        embeddings = self.elmo (character_ids)
        elmo_output = embeddings['elmo_representations'][0]
        
        # Average pool over sequence
        pooled = elmo_output.mean (dim=1)  # (batch, 1024)
        
        # Classify
        logits = self.classifier (pooled)
        return logits

# Initialize classifier
classifier = TextClassifier (elmo, num_classes=2)

# Forward pass
logits = classifier (character_ids)
print(f"Logits shape: {logits.shape}")  # (2, 2) - batch_size × num_classes
\`\`\`

## Comparing Static vs Contextualized Embeddings

\`\`\`python
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# Train simple Word2Vec for comparison
sentences_train = [
    ['I', 'deposited', 'money', 'bank'],
    ['river', 'bank', 'nature', 'water'],
    ['financial', 'bank', 'account'],
] * 100

w2v = Word2Vec (sentences_train, vector_size=100, window=5, min_count=1, sg=1)

# Static embedding: same vector always
bank_static = w2v.wv['bank']
print(f"Static embedding (Word2Vec): same vector always")
print(f"Shape: {bank_static.shape}")

# Contextualized: different vectors per context
# (Using ELMo as shown above)
# bank_financial = ELMo("I deposited at the bank")
# bank_river = ELMo("We sat on the river bank")

# Similarity comparison
print("\\n=== Similarity Analysis ===")

# With static embeddings:
# "bank" has same vector in all contexts
# Similarity to "financial" and "river" is fixed

# With contextualized embeddings:
# "bank" in financial context → high similarity to "deposit", "money"
# "bank" in river context → high similarity to "river", "water"
# "bank" vectors themselves have LOW similarity across contexts

print("Static: 'bank' is same in all contexts")
print("Contextualized: 'bank' changes based on surrounding words")
\`\`\`

## Why Contextualized Embeddings Work Better

### 1. Disambiguation

\`\`\`python
# Example: "play"

contexts = [
    "The play was excellent",      # Theatre performance
    "Children play in the park",   # Activity
    "He will play guitar",         # Musical performance
    "The play button is broken",   # Control mechanism
]

# Static embedding: ONE vector for all meanings
# Contextualized: FOUR different vectors, each specialized
\`\`\`

### 2. Handling Syntax

\`\`\`python
# Same word, different grammatical roles

"The chicken is ready to eat"
# chicken = food (object being eaten)

"The chicken is ready to eat the corn"
# chicken = animal (subject doing eating)

# Contextualized embeddings capture this syntactic difference
# Static embeddings cannot
\`\`\`

### 3. Capturing Nuance

\`\`\`python
sentences = [
    "This is good",           # Positive
    "This is not good",       # Negative (negation)
    "This is very good",      # More positive (intensifier)
    "This is good, I guess",  # Hedged positive (uncertainty)
]

# "good" gets different contextual embeddings
# Capturing sentiment modifiers automatically
\`\`\`

## Practical Benefits

### Downstream Task Performance

Studies show ELMo improvements:
- **Question Answering**: +4-6% F1 score
- **Named Entity Recognition**: +2-4% F1 score
- **Sentiment Analysis**: +3-5% accuracy
- **Textual Entailment**: +5-8% accuracy

### Transfer Learning

\`\`\`python
# Pre-train on large corpus (Wikipedia, news)
# → Learns general linguistic knowledge

# Fine-tune or use frozen for specific task
# → Transfers knowledge to your domain

# Benefits:
# - Better with limited labeled data
# - Captures linguistic patterns
# - Reduces training time
\`\`\`

## ELMo vs Word2Vec: Side-by-Side

| Aspect | Word2Vec | ELMo |
|--------|----------|------|
| **Embedding Type** | Static | Contextualized |
| **One Word** | One vector always | Different vectors per context |
| **Polysemy** | Cannot handle | Naturally handles |
| **Architecture** | Shallow (single layer) | Deep (bidirectional LSTM) |
| **Training** | Unsupervised (co-occurrence) | Unsupervised (language model) |
| **OOV** | Cannot handle (except FastText) | Handles via char CNN |
| **Dimensions** | 100-300 | 1024+ |
| **Speed** | Very fast | Slower (LSTM) |
| **Memory** | Low | Higher |
| **Quality** | Good for static tasks | Better for complex tasks |

## Limitations and Future Directions

### ELMo Limitations

1. **Computational Cost**: LSTM slower than transformers
2. **Sequential Processing**: Cannot parallelize like transformers
3. **Limited Context**: LSTM memory limited compared to attention
4. **Architecture**: RNN-based, less efficient than attention

### The Path to Transformers

ELMo demonstrated contextualized embeddings work, but:
- **BERT (2018)**: Uses transformers instead of LSTMs → faster, better
- **GPT (2018)**: Transformer-based language model → more scalable
- **Modern models**: All use attention, not RNNs

**ELMo\'s Legacy:**
- Proved context-dependent embeddings essential
- Popularized pre-training + fine-tuning
- Sparked transformer revolution
- Foundation for modern NLP

## Implementation Best Practices

\`\`\`python
# Best practices for using contextualized embeddings

# 1. Choose the right model
# - ELMo: Good baseline, handles OOV
# - BERT: Better quality, more widely used
# - GPT: Good for generation tasks

# 2. Use appropriate layers
# Lower layers: Syntax, POS tags
# Middle layers: Semantic meaning
# Higher layers: Task-specific features

# 3. Fine-tune when possible
# Frozen embeddings: Faster, less data needed
# Fine-tuned: Better performance, more data needed

# 4. Consider computational constraints
# ELMo: ~1024 dims, LSTM inference
# BERT: ~768 dims, transformer inference
# Choose based on latency/accuracy tradeoff

# 5. Batch processing for efficiency
# Process multiple sentences together
# Utilize GPU parallelization
\`\`\`

## Summary

**Key Takeaways:**
1. Contextualized embeddings solve polysemy with context-dependent vectors
2. ELMo uses bidirectional LSTMs to generate dynamic representations
3. Same word in different contexts gets different embeddings
4. Dramatically improves downstream task performance
5. Pre-training captures general linguistic knowledge

**Evolution:**
- Static embeddings (Word2Vec, 2013) → One vector per word
- Contextualized embeddings (ELMo, 2018) → Dynamic vectors
- Transformers (BERT, 2018) → Attention-based contextualized embeddings
- Modern NLP (2024) → All use contextualized representations

**Next Section:**
We'll explore sequence modeling with RNNs and LSTMs, understanding how they process sequential text data—the foundation that ELMo builds upon.
`,
};
