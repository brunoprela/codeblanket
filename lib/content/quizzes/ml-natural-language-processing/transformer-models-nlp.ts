import { QuizQuestion } from '../../../types';

export const transformerModelsNlpQuiz: QuizQuestion[] = [
  {
    id: 'transformer-models-nlp-dq-1',
    question:
      'Compare BERT and GPT architectures. Explain why BERT is bidirectional while GPT is unidirectional, and how this affects their suitability for different NLP tasks.',
    sampleAnswer: `BERT and GPT represent fundamentally different design philosophies:

**BERT (Bidirectional Encoder):**
- Uses only encoder stack
- Bidirectional: sees full context (left and right)
- Pre-training: Masked Language Modeling (MLM)
- Cannot generate text naturally

**GPT (Unidirectional Decoder):**
- Uses only decoder stack
- Unidirectional: sees only left context
- Pre-training: Causal Language Modeling
- Natural for text generation

**Why the Difference:**

BERT's bidirectionality comes from MLM:
- Input: "The [MASK] sat on the mat"
- Model sees: "The" and "sat on the mat"
- Uses both sides to predict "cat"
- Requires seeing full sentence

GPT's unidirectionality from autoregressive generation:
- Predicts next token given previous tokens
- Cannot see future (doesn't exist yet)
- Left-to-right only

**Task Suitability:**

BERT excels at understanding:
1. **Classification**: "Is this positive?" - needs full sentence
2. **NER**: "Is 'Apple' a company?" - needs context from both sides
3. **QA**: Find answer span - bidirectional matching
4. **Similarity**: Compare sentence meanings

GPT excels at generation:
1. **Text completion**: Natural autoregressive task
2. **Creative writing**: Generate stories
3. **Dialogue**: Response generation
4. **Summarization**: Generate summaries

**Can GPT do understanding?** Yes, with prompting:
- "Sentiment: 'Great movie!' Answer:"
- Not as efficient as BERT for understanding

**Can BERT generate?** Poorly:
- No causal structure for sequential generation
- Would need iterative masked prediction`,
    keyPoints: [
      'BERT: bidirectional encoder, masked LM, best for understanding',
      'GPT: unidirectional decoder, causal LM, best for generation',
      'Bidirectionality requires seeing full input (BERT)',
      'Generation requires left-to-right processing (GPT)',
      'Task determines architecture: understanding→BERT, generation→GPT',
    ],
  },
  {
    id: 'transformer-models-nlp-dq-2',
    question:
      'Explain why positional encodings are necessary in transformers, and how they differ from the position information in RNNs/LSTMs.',
    sampleAnswer: `Positional encodings are critical because transformers have no inherent notion of token order:

**Why Necessary:**

Attention mechanism is permutation-invariant:
\`\`\`
Attention("cat sat", K, V) = Attention("sat cat", K, V)
\`\`\`

Without positional information:
- "dog bites man" = "man bites dog"
- Word order lost
- Grammar and meaning destroyed

**RNN vs Transformer Position:**

**RNNs/LSTMs (Implicit):**
- Process sequentially: word₁ → h₁ → word₂ → h₂
- Position encoded in processing order
- Earlier words affect later hidden states
- Position information is built-in

**Transformers (Explicit):**
- Process all positions simultaneously
- No inherent order
- Must explicitly add position information
- Positional encoding added to embeddings

**Implementation:**

Sinusoidal positional encoding:
\`\`\`python
PE(pos, 2i) = sin (pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos (pos / 10000^(2i/d_model))
\`\`\`

Properties:
- Unique encoding for each position
- Smooth relative positions
- Can extrapolate to unseen lengths
- No learned parameters

**Alternative: Learned Positional Embeddings:**
\`\`\`python
self.pos_embedding = nn.Embedding (max_len, d_model)
\`\`\`

Trade-offs:
- Sinusoidal: Can handle any length, fixed
- Learned: Better fit to data, limited length

**Why This Matters:**

Without positional encoding:
- Grammar destroyed
- "Not good" = "good not" (different meanings!)
- Sequential dependencies lost

With positional encoding:
- Word order preserved
- Syntax and grammar maintained
- Model learns position-dependent patterns`,
    keyPoints: [
      'Attention is permutation-invariant, position information lost',
      'RNNs: implicit position from sequential processing',
      'Transformers: explicit positional encodings required',
      'Sinusoidal encodings: unique per position, extrapolate to any length',
      'Without positions: word order lost, grammar destroyed',
    ],
  },
  {
    id: 'transformer-models-nlp-dq-3',
    question:
      'Transformers achieve state-of-the-art results but are computationally expensive. Discuss the computational trade-offs and describe strategies to make transformers more efficient for production use.',
    sampleAnswer: `Transformers have high computational costs but offer strategies for efficiency:

**Computational Costs:**

1. **Self-attention complexity: O(n²d)**
   - n = sequence length
   - d = model dimension
   - Quadratic in sequence length!

2. **Memory requirements:**
   - BERT-base: 110M parameters
   - GPT-3: 175B parameters
   - Attention matrices: O(n²) memory

3. **Inference latency:**
   - Multiple attention layers (12-24)
   - Large matrix multiplications
   - ~100-500ms per query

**Efficiency Strategies:**

**1. Model Distillation:**
\`\`\`python
# DistilBERT: 40% smaller, 60% faster
# Retains 97% performance
from transformers import DistilBertModel

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 66M vs 110M parameters
# Inference: 2-3x faster
\`\`\`

**2. Quantization:**
\`\`\`python
# 8-bit or 4-bit quantization
# Reduces model size by 4-8x
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('bert-base')
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 110MB → 28MB
\`\`\`

**3. Pruning:**
- Remove less important attention heads
- Prune low-magnitude weights
- 30-50% reduction with <1% accuracy loss

**4. Efficient Attention:**
- Linformer: O(n) complexity
- Performer: Linear attention
- Longformer: Sparse attention patterns

**5. Caching:**
\`\`\`python
# Cache key-value pairs for generation
past_key_values = None
for token in tokens:
    output = model (token, past_key_values=past_key_values)
    past_key_values = output.past_key_values
# Speeds up generation significantly
\`\`\`

**6. Smaller Models:**
- BERT-tiny: 4.4M parameters
- ALBERT: Parameter sharing
- MobileBERT: Optimized for mobile

**Production Strategy:**

\`\`\`python
# 1. Choose right model for task
task_to_model = {
    'simple_classification': 'distilbert-base',
    'complex_understanding': 'roberta-large',
    'real_time': 'albert-base-v2',
    'edge_device': 'distilbert-base' + quantization
}

# 2. Optimize inference
# - Batch requests
# - Use ONNX runtime
# - GPU when available, quantized CPU otherwise

# 3. Cache common queries
from functools import lru_cache

@lru_cache (maxsize=1000)
def predict (text):
    return model (text)

# 4. Monitor latency/cost
# - Set timeout limits
# - Fallback to simpler models
# - A/B test model sizes
\`\`\`

**Real-World Trade-offs:**

| Model | Size | Latency | Accuracy | Use Case |
|-------|------|---------|----------|----------|
| BERT-large | 340MB | 200ms | 95% | Offline batch |
| BERT-base | 110MB | 100ms | 93% | General API |
| DistilBERT | 66MB | 50ms | 91% | Real-time API |
| BERT-tiny | 17MB | 20ms | 85% | Edge devices |

**Recommendation:**

Start with distilled model:
- 90%+ accuracy retention
- 2-3x faster
- Easier deployment
- Upgrade if accuracy insufficient`,
    keyPoints: [
      'Transformers: O(n²) attention complexity, large memory',
      'Distillation: 40% smaller, 60% faster, 97% performance (DistilBERT)',
      'Quantization: 4-8x smaller model size',
      'Efficient attention variants: Linformer, Longformer reduce complexity',
      'Production: use distilled models, quantization, caching',
      'Trade-off: size/speed vs accuracy based on requirements',
    ],
  },
];
