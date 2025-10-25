/**
 * Section: Transformer Models for NLP
 * Module: Natural Language Processing
 *
 * Covers BERT, GPT, and transformer architecture for NLP
 */

export const transformerModelsNlpSection = {
  id: 'transformer-models-nlp',
  title: 'Transformer Models for NLP',
  content: `
# Transformer Models for NLP

## Introduction

Transformers revolutionized NLP by replacing recurrent architectures with pure attention mechanisms. BERT and GPT became the foundation for virtually all modern NLP, achieving state-of-the-art results across tasks.

**Key Innovation:**
- **Pure attention**: No recurrence, full parallelization
- **Bidirectional context**: BERT sees full context
- **Scalability**: Trains efficiently on massive datasets
- **Transfer learning**: Pre-train once, fine-tune for many tasks

## Transformer Architecture

### Core Components

\`\`\`python
import torch
import torch.nn as nn
import math

class TransformerBlock (nn.Module):
    """Single Transformer Block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention (d_model, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear (d_model, d_ff),
            nn.GELU(),
            nn.Linear (d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm (d_model)
        self.norm2 = nn.LayerNorm (d_model)
        
        # Dropout
        self.dropout = nn.Dropout (dropout)
        
    def forward (self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention (x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout (attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff (x)
        x = self.norm2(x + self.dropout (ff_output))
        
        return x
\`\`\`

### Positional Encoding

\`\`\`python
class PositionalEncoding (nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp (torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin (position * div_term)
        pe[:, 1::2] = torch.cos (position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward (self, x):
        return x + self.pe[:, :x.size(1)]
\`\`\`

## BERT: Bidirectional Encoder

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder and masked language modeling.

### BERT Architecture

\`\`\`python
from transformers import BertModel, BertTokenizer

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
text = "BERT revolutionized NLP"
tokens = tokenizer (text, return_tensors='pt')

# Forward pass
outputs = model(**tokens)

# Get embeddings
last_hidden_states = outputs.last_hidden_state  # (1, seq_len, 768)
pooled_output = outputs.pooler_output  # (1, 768) - [CLS] token

print(f"Hidden states shape: {last_hidden_states.shape}")
print(f"Pooled output shape: {pooled_output.shape}")
\`\`\`

### BERT Pre-training

\`\`\`python
# Masked Language Modeling (MLM)
# Randomly mask 15% of tokens, predict them

text = "The cat sat on the [MASK]"
inputs = tokenizer (text, return_tensors='pt')

# Model predicts masked tokens
outputs = model(**inputs)
predictions = outputs.last_hidden_state

# Task 2: Next Sentence Prediction (NSP)
sentence_a = "The cat sat on the mat"
sentence_b = "The dog ran in the park"

# Model predicts if sentence_b follows sentence_a
inputs = tokenizer (sentence_a, sentence_b, return_tensors='pt')
outputs = model(**inputs)
\`\`\`

### Using BERT for Classification

\`\`\`python
from transformers import BertForSequenceClassification

class BERTClassifier (nn.Module):
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout (dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward (self, input_ids, attention_mask):
        outputs = self.bert (input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        pooled_output = self.dropout (pooled_output)
        logits = self.classifier (pooled_output)
        return logits

# Training example
model = BERTClassifier (num_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
for batch in dataloader:
    input_ids, attention_mask, labels = batch
    
    optimizer.zero_grad()
    logits = model (input_ids, attention_mask)
    loss = criterion (logits, labels)
    loss.backward()
    optimizer.step()
\`\`\`

## GPT: Generative Pre-trained Transformer

GPT uses decoder-only architecture with causal masking for text generation.

### GPT Architecture

\`\`\`python
from transformers import GPT2Model, GPT2Tokenizer

# Load pre-trained GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Tokenize
text = "Natural language processing"
tokens = tokenizer (text, return_tensors='pt')

# Forward pass
outputs = model(**tokens)
hidden_states = outputs.last_hidden_state

print(f"Hidden states shape: {hidden_states.shape}")
\`\`\`

### Text Generation with GPT

\`\`\`python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text (prompt, max_length=50, temperature=1.0, top_k=50):
    """Generate text autoregressively"""
    input_ids = tokenizer.encode (prompt, return_tensors='pt')
    
    # Generate
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode (output[0], skip_special_tokens=True)
    return generated_text

# Example
prompt = "Machine learning is"
generated = generate_text (prompt, max_length=30)
print(f"Generated: {generated}")
\`\`\`

## BERT vs GPT Comparison

\`\`\`
BERT (Encoder):
- Bidirectional context
- Masked language modeling
- Best for understanding tasks
- Cannot generate text naturally

GPT (Decoder):
- Unidirectional (left-to-right)
- Causal language modeling
- Best for generation tasks
- Can be used for understanding too

Architecture:
BERT: Input → Encoder → Output
GPT: Input → Decoder → Next token
\`\`\`

### BERT for Understanding

\`\`\`python
# Sentence classification
from transformers import pipeline

classifier = pipeline('sentiment-analysis', model='bert-base-uncased')
result = classifier("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]

# Question answering
qa = pipeline('question-answering', model='bert-base-uncased')
context = "Paris is the capital of France"
question = "What is the capital of France?"
answer = qa (question=question, context=context)
print(answer)  # {'answer': 'Paris', 'score': 0.987}
\`\`\`

### GPT for Generation

\`\`\`python
# Text generation
generator = pipeline('text-generation', model='gpt2')
result = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])

# Completion
prompt = "The future of AI is"
result = generator (prompt, max_length=30)
print(result[0]['generated_text'])
\`\`\`

## Modern Transformer Variants

### RoBERTa (Robustly Optimized BERT)

\`\`\`python
from transformers import RobertaModel

# Improvements over BERT:
# - No NSP task
# - Dynamic masking
# - Larger batches
# - More data

model = RobertaModel.from_pretrained('roberta-base')
\`\`\`

### DistilBERT (Distilled BERT)

\`\`\`python
from transformers import DistilBertModel

# 40% smaller, 60% faster, 97% performance
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
\`\`\`

### T5 (Text-to-Text Transfer Transformer)

\`\`\`python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Everything as text-to-text
input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer (input_text, return_tensors='pt').input_ids

outputs = model.generate (input_ids)
print(tokenizer.decode (outputs[0]))  # "Bonjour, comment allez-vous?"
\`\`\`

## Best Practices

### 1. Model Selection

\`\`\`python
# Choose based on task:
tasks = {
    'classification': 'bert-base-uncased',
    'generation': 'gpt2',
    'general': 't5-base',
    'speed': 'distilbert-base-uncased',
    'accuracy': 'roberta-large'
}
\`\`\`

### 2. Fine-tuning Strategy

\`\`\`python
# Learning rate schedule
from transformers import get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=10000
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
\`\`\`

### 3. Efficient Training

\`\`\`python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale (loss).backward()
    scaler.step (optimizer)
    scaler.update()
\`\`\`

## Summary

Transformers are the foundation of modern NLP:
- **BERT**: Bidirectional, understanding tasks
- **GPT**: Unidirectional, generation tasks
- **Variants**: Optimized for speed, size, or accuracy
- **Key**: Pre-training + fine-tuning paradigm

**Next**: Fine-tuning transformers for specific tasks.
`,
};
