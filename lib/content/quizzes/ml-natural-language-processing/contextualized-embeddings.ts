import { QuizQuestion } from '../../../types';

export const contextualizedEmbeddingsQuiz: QuizQuestion[] = [
  {
    id: 'contextualized-embeddings-dq-1',
    question:
      'Explain how ELMo solves the polysemy problem that static word embeddings (Word2Vec, GloVe) cannot address. Use the word "bank" as an example to illustrate the mechanism and benefits.',
    sampleAnswer: `ELMo solves polysemy by generating context-dependent embeddings dynamically, rather than assigning one fixed vector per word type.

**The Polysemy Problem:**

Static embeddings assign ONE vector to each word regardless of context:
- "bank" → [0.23, -0.15, 0.87, ...] (always this vector)

This single vector must represent:
1. Financial institution: "I deposited money at the bank"
2. River edge: "We sat on the river bank"
3. Blood bank: "She donated to the blood bank"

The resulting vector averages all contexts, poorly representing any specific meaning.

**How ELMo Solves This:**

ELMo uses bidirectional LSTMs that read the entire sentence to generate embeddings:

Sentence 1: "I deposited money at the bank"
- Forward LSTM: reads "I deposited money at the" → hidden state
- Backward LSTM: reads "(end) bank the at money deposited" → hidden state
- Combines bidirectional context around "bank"
- Sees words like "deposited", "money" → financial context
- Generates embedding specialized for financial meaning

Sentence 2: "We sat on the river bank"
- Forward LSTM: reads "We sat on the river" → different hidden state
- Backward LSTM: reads "(end) bank river the on sat" → different hidden state
- Sees words like "river", "sat" → geographical context
- Generates DIFFERENT embedding specialized for geographical meaning

**The Mechanism:**

\`\`\`
ELMo Architecture:

Input: "I deposited money at the bank"
       ↓ ↓ ↓ ↓ ↓ ↓ ↓
Character CNN (handles each word)
       ↓ ↓ ↓ ↓ ↓ ↓ ↓
BiLSTM Layer 1 →→→→→→→ (forward context)
                ←←←←←← (backward context)
       ↓ ↓ ↓ ↓ ↓ ↓ ↓
BiLSTM Layer 2 →→→→→→→
                ←←←←←←
       ↓ ↓ ↓ ↓ ↓ ↓ ↓
For "bank": weighted combination of layers
       ↓
Context-specific embedding
\`\`\`

**Key Benefits:**

1. **Automatic Disambiguation:**
   - No manual sense annotation needed
   - Model learns from surrounding words
   - Different contexts → different vectors automatically

2. **Bidirectional Context:**
   - Forward LSTM: "deposited money at the" suggests financial
   - Backward LSTM: starts from end, also sees "money", "deposited"
   - Combines both directions for robust understanding

3. **Layered Representations:**
   - Layer 0 (char CNN): Morphology, spelling
   - Layer 1 (LSTM 1): Syntax, POS tags
   - Layer 2 (LSTM 2): Semantics, meaning
   - Weighted combination captures multi-level information

4. **Empirical Results:**
   Measuring similarity between "bank" embeddings:
   - Static (Word2Vec): "bank" vs "bank" similarity = 1.0 (always identical)
   - ELMo: "bank" (financial) vs "bank" (river) similarity ≈ 0.4-0.6
   - Lower similarity correctly reflects different meanings!

**Downstream Impact:**

For Named Entity Recognition:
- "Apple announced new iPhone" 
  - Static: "apple" vector includes fruit meaning → confusion
  - ELMo: Sees "announced", "iPhone" → generates organization-context embedding
  
For Sentiment Analysis:
- "This bank is solid" (financial → positive)
- "The bank is muddy" (river → neutral/negative)
- ELMo disambiguates, improving classification

**Practical Advantage:**

Production scenario: Financial NLP system trained on banking texts
- Test input: "They cleaned the river bank"
- Static embedding: "bank" has strong financial associations → false positive
- ELMo: Reads "river" context → generates geographical embedding → correct classification

**Why Static Embeddings Can't Do This:**

Static embeddings are computed ONCE during training:
1. Scan corpus, count co-occurrences
2. Factor matrix or train skip-gram
3. Store one vector per word
4. At inference: lookup same vector always

ELMo computes embeddings DYNAMICALLY at inference:
1. Receive sentence
2. Run through BiLSTM
3. Generate context-specific embeddings
4. Different sentence → different embeddings

This dynamic computation is what enables disambiguation.`,
    keyPoints: [
      'Static embeddings: one vector per word type, averages all contexts',
      'ELMo: dynamic embeddings generated from bidirectional LSTM reading full sentence',
      'Bidirectional context captures meaning from both directions',
      'Same word in different contexts gets different vectors',
      'Automatic disambiguation without manual sense annotation',
      'Empirically shows lower similarity for different senses vs static 1.0',
    ],
  },
  {
    id: 'contextualized-embeddings-dq-2',
    question:
      'ELMo uses a bidirectional LSTM architecture. Why is bidirectionality important for generating high-quality contextualized embeddings? Provide examples where forward-only context would fail.',
    sampleAnswer: `Bidirectionality is crucial because meaning often depends on words appearing AFTER the target word, not just before. Forward-only models would miss critical context.

**The Problem with Forward-Only:**

Forward-only LSTMs process text left-to-right, seeing only preceding context:

\`\`\`
"The animal didn't cross the street because it was too tired"
                                                    ↑
When embedding "it", forward LSTM sees:
"The animal didn't cross the street because it was too"
Cannot see "tired" → cannot resolve pronoun reference
\`\`\`

**Example 1: Pronoun Resolution**

Sentence: "The trophy didn't fit in the suitcase because it was too big"

Forward-only at "it":
- Sees: "The trophy didn't fit in the suitcase because it was too"
- Cannot see "big" yet
- Ambiguous: does "it" refer to trophy or suitcase?

Bidirectional:
- Forward: "The trophy didn't fit in the suitcase because it was too"
- Backward: "big too was it because suitcase the in fit didn't trophy The"
- Sees "big" from backward pass
- Disambiguates: "big" → likely "trophy" (trophy too big for suitcase)

**Example 2: Sentiment with End-of-Sentence Modifiers**

"The movie was good, I guess"
                          ↑
Forward-only at "good":
- Sees: "The movie was"
- Predicts positive sentiment
- Misses "I guess" which hedges/weakens positivity

Bidirectional:
- Backward pass sees "I guess" 
- Adjusts "good" embedding to reflect hedged/uncertain sentiment
- More accurate for sentiment analysis

**Example 3: Negation Appearing After**

"I thought the food would be terrible, but it was not"
                                                   ↑
Forward-only at "not":
- Sees "terrible" earlier, might embed negative context
- But "not" is negating something that comes conceptually from prior expectation

Bidirectional:
- Can integrate full sentence context
- Understands contrastive structure
- "terrible" + "but" + "not" → actually positive

**Example 4: Word Sense Disambiguation**

"She went to the bank"

Forward-only:
- Sees: "She went to the"
- Could be: river bank, financial bank, blood bank, etc.
- Insufficient context

"She went to the bank to deposit money"

Forward-only at "bank":
- Still only sees "She went to the"
- Hasn't reached "deposit money" yet
- Cannot disambiguate

Bidirectional:
- Backward pass sees "deposit money"
- Forward pass sees "She went to"
- Combines: "went to" + "deposit money" → financial bank
- Correct disambiguation

**Example 5: Syntax-Dependent Meaning**

"The chicken is ready to eat"

Forward-only at "chicken":
- Sees: "The"
- No way to know if chicken is subject (eating) or object (being eaten)

Full sentence context needed:
- "The chicken is ready to eat" (chicken = food, object)
- "The chicken is ready to eat the corn" (chicken = animal, subject)

Bidirectional:
- Backward pass sees what follows
- Disambiguates grammatical role and meaning

**The Mechanism:**

ELMo\'s bidirectional architecture:

\`\`\`
Input: "I deposited money at the bank"

Forward LSTM:
h₁(I) → h₂(deposited) → h₃(money) → h₄(at) → h₅(the) → h₆(bank)
                                                          ↑
                                                 Uses context: I deposited money at the

Backward LSTM:
h₆(bank) ← h₅(the) ← h₄(at) ← h₃(money) ← h₂(deposited) ← h₁(I)
    ↑
Uses context: (end of sentence, could include more)

Final embedding for "bank":
  = combine (forward_h₆, backward_h₆)
  = context from both directions
\`\`\`

**Why Both Directions Matter:**

1. **Syntactic Structure:**
   - Forward: Subject-verb patterns
   - Backward: Object-verb patterns
   - Both: Complete parse understanding

2. **Long-Range Dependencies:**
   - Forward: Anaphora (pronoun refers to earlier noun)
   - Backward: Cataphora (pronoun refers to later noun)
   - Both: All reference resolution

3. **Semantic Ambiguity:**
   - Forward: Priming effects, expectations
   - Backward: Confirmation, resolution
   - Both: Full context disambiguation

**Empirical Evidence:**

Studies comparing forward-only vs bidirectional:
- Bidirectional ELMo: State-of-art performance
- Forward-only ELMo: -3 to -5% accuracy on most tasks
- Particularly large gap on:
  - Coreference resolution
  - Reading comprehension
  - Tasks requiring full sentence understanding

**Modern Architectures:**

This insight led to:
- BERT: Bidirectional Encoder Representations from Transformers
- Explicitly bidirectional design
- Even better than LSTM-based ELMo

GPT (forward-only) vs BERT (bidirectional):
- GPT: Excellent for generation (predict next word)
- BERT: Better for understanding (uses full context)

**Key Insight:**

Natural language meaning is not purely left-to-right. Humans read entire sentences and integrate context from both directions to understand meaning. Bidirectional models mimic this process computationally, achieving better language understanding by leveraging full sentential context.`,
    keyPoints: [
      'Forward-only sees only preceding context, misses crucial following words',
      'Bidirectional reads both directions: forward + backward context',
      'Critical for pronoun resolution, negation, end-of-sentence modifiers',
      'Examples: "it was too tired" needs "tired", "good, I guess" needs "I guess"',
      'Bidirectional ELMo outperforms forward-only by 3-5% on most tasks',
      'Led to BERT design which is explicitly bidirectional',
    ],
  },
  {
    id: 'contextualized-embeddings-dq-3',
    question:
      'ELMo marked a transition point in NLP before the transformer revolution. Compare ELMo to modern transformer-based models (BERT). What advantages did ELMo demonstrate, and what limitations led to transformers becoming dominant?',
    sampleAnswer: `ELMo was a crucial stepping stone that proved contextualized embeddings work, but transformers (BERT) overcame its architectural limitations.

**ELMo\'s Groundbreaking Contributions:**

1. **Proved Contextualized Embeddings Work:**
   - First widely-adopted contextualized embedding
   - Showed dramatic improvements over static embeddings
   - +4-6% across many NLP tasks
   - Established pre-training + fine-tuning paradigm

2. **Handled Polysemy:**
   - Same word, different contexts → different embeddings
   - Solved decades-old problem in computational linguistics
   - No manual sense annotation required

3. **Transfer Learning:**
   - Pre-train on large corpus (1 Billion Word Benchmark)
   - Transfer to downstream tasks
   - Worked well with limited labeled data

4. **Character-Based:**
   - Character CNN handled OOV words
   - Robust to typos and morphological variation

**ELMo's Architectural Limitations:**

1. **Sequential Processing (LSTMs):**

\`\`\`
LSTM processes words one-by-one:
word₁ → h₁ → word₂ → h₂ → word₃ → h₃ → ...

Cannot parallelize across sequence:
- Must wait for h₁ before computing h₂
- Must wait for h₂ before computing h₃
- Sequential bottleneck
\`\`\`

Impact:
- Slow training and inference
- Cannot utilize GPU parallelism effectively
- Scales poorly to longer sequences

2. **Limited Context Window:**

LSTMs suffer from:
- Vanishing gradients over long sequences
- Effectively ~100-200 token context window
- Struggles with document-level understanding

Example limitation:
\`\`\`
Document with 500 words
ELMo: Each word's embedding influenced mostly by nearby words
       Distant context (300 words away) has diminishing impact
BERT: Full document attention (up to 512 tokens at once)
      Every word attends to every other word
\`\`\`

3. **Bidirectional But Not Truly Symmetric:**

ELMo:
\`\`\`
Forward LSTM: Left-to-right
Backward LSTM: Right-to-left
Concatenate: [forward; backward]
\`\`\`

Two separate models, combined post-hoc
Not jointly optimized on bidirectional objective

4. **Computational Inefficiency:**

- 2 separate LSTMs (forward + backward)
- Each LSTM step depends on previous step
- No attention mechanism (all context funneled through hidden state)

**How BERT (Transformers) Improved:**

1. **Parallel Processing:**

\`\`\`
Transformer (BERT):
All positions processed simultaneously using attention:

word₁ ───┐
word₂ ───┼──→ Attention layer → All embeddings computed in parallel
word₃ ───┤
...   ───┘

Massive speedup:
- Can utilize full GPU parallelism
- Training time: weeks (ELMo) → days (BERT)
\`\`\`

2. **True Bidirectionality:**

BERT's masked language modeling:
\`\`\`
Input: "I deposited [MASK] at the bank"
Model must predict: "money"

Uses both left and right context simultaneously:
- "I deposited" (left)
- "at the bank" (right)
- Jointly optimizes for bidirectional understanding
\`\`\`

vs ELMo: two separate unidirectional models

3. **Self-Attention Mechanism:**

\`\`\`
LSTM:
- Context funneled through hidden state bottleneck
- Distant words harder to attend to

Transformer:
- Every word directly attends to every other word
- No distance decay
- Explicit attention weights show what model focuses on
\`\`\`

4. **Longer Context:**

- ELMo: Effective ~100-200 tokens
- BERT-base: 512 tokens
- BERT-large: 512 tokens
- Modern transformers: 2048-100K+ tokens

5. **Better Scaling:**

Transformers scale better with:
- Model size (parameters)
- Data size (corpus)
- Compute (GPUs/TPUs)

Empirical scaling laws favor transformers.

**Performance Comparison:**

Benchmark results (relative improvements):

| Task | Word2Vec | ELMo | BERT |
|------|----------|------|------|
| SQuAD (QA) | 70% | 85% (+15%) | 93% (+23%) |
| MNLI (NLI) | 72% | 78% (+6%) | 86% (+14%) |
| NER (F1) | 88% | 92% (+4%) | 95% (+7%) |

ELMo was major improvement, BERT was even larger improvement.

**Why Transformers Won:**

1. **Speed:** 10-100x faster training due to parallelization
2. **Quality:** Better performance on virtually all tasks
3. **Scalability:** Can train much larger models effectively
4. **Flexibility:** Same architecture for many tasks
5. **Interpretability:** Attention weights provide insights

**ELMo\'s Lasting Impact:**

Despite BERT dominance, ELMo's legacy:
- **Proved contextualized embeddings essential** (now standard)
- **Established pre-training paradigm** (used by all modern models)
- **Showed transfer learning works in NLP** (now universal)
- **Inspired BERT** (authors explicitly built on ELMo's ideas)

**When to Use ELMo Today:**

ELMo still useful for:
- **Resource-constrained environments:** Smaller than BERT
- **Character-level robustness:** Built-in char CNN
- **Legacy systems:** Already integrated
- **Education:** Simpler architecture, easier to understand

But for new projects: use BERT or newer transformers.

**Historical Timeline:**

2013: Word2Vec (static embeddings)
↓
2018: ELMo (contextualized, LSTM-based)
↓
2018: BERT (contextualized, transformer-based) ← Current paradigm
↓
2018-2024: GPT, T5, BART, RoBERTa, etc. (all transformers)

**Key Insight:**

ELMo was the critical bridge between static embeddings and transformers. It proved contextualized representations were essential and established the pre-training paradigm, but its LSTM architecture had fundamental limitations. Transformers solved these limitations with parallel attention, becoming the dominant architecture. Every modern NLP model (GPT, BERT, T5, etc.) uses transformers, not LSTMs.`,
    keyPoints: [
      'ELMo proved contextualized embeddings work, +4-6% over static embeddings',
      'LSTM architecture: sequential processing, cannot parallelize, slow',
      'Limited context window (~100-200 tokens) due to LSTM vanishing gradients',
      'BERT (transformers): parallel processing, true bidirectional, self-attention',
      'Transformers 10-100x faster training, better performance, scale better',
      "ELMo\'s legacy: established contextualized embeddings and pre-training paradigm",
      'Modern NLP: all use transformers (BERT, GPT, T5), not LSTMs',
    ],
  },
];
