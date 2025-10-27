import { QuizQuestion } from '../../../types';

export const attentionNlpQuiz: QuizQuestion[] = [
  {
    id: 'attention-nlp-dq-1',
    question:
      'Explain the three components of attention (Query, Key, Value) and how they work together in the attention mechanism. Use a concrete NLP example to illustrate.',
    sampleAnswer: `The Query-Key-Value framework enables dynamic focus on relevant information:

**Components:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I offer?"
- **Value (V)**: "My actual content"

**Mechanism:**1. Compare Query with all Keys (similarity scores)
2. Apply softmax to get attention weights
3. Weight the Values by attention scores
4. Sum to get context-aware output

**Concrete Example: Machine Translation**

Translating "I love cats" → "J'aime les chats"

When generating "chats" (French for cats):
- **Query**: Decoder state representing "what word to generate now"
- **Keys**: All encoder states from source sentence
- **Values**: Same encoder states (contain actual information)

Process:
1. Compare Query("chats") with Key("I"): low score
2. Compare Query("chats") with Key("love"): low score  
3. Compare Query("chats") with Key("cats"): HIGH score
4. Softmax: [0.1, 0.1, 0.8]
5. Output = 0.1×Value("I") + 0.1×Value("love") + 0.8×Value("cats")

The model learns to attend strongly to "cats" when generating "chats".

**Why This Works:**
- Dynamically focuses on relevant parts
- Learns alignment between source and target
- No fixed window size
- Interpretable attention weights`,
    keyPoints: [
      'Query: what to look for, Key: what offered, Value: actual content',
      'Attention = softmax(Query·Key) applied to Values',
      'Enables dynamic focus on relevant information',
      'Example: translating "cats"→"chats" attends to source "cats"',
      'Interpretable: can visualize attention weights',
    ],
  },
  {
    id: 'attention-nlp-dq-2',
    question:
      'Why does multi-head attention use multiple attention mechanisms in parallel rather than a single larger attention? What different aspects might different heads capture?',
    sampleAnswer: `Multi-head attention allows the model to attend to different types of information simultaneously, each head specializing in different patterns.

**Why Multiple Heads:**

Single large attention would learn one blended representation. Multiple heads allow specialized focuses:
- Head 1: Syntactic relationships
- Head 2: Semantic associations  
- Head 3: Long-range dependencies
- Head 4: Positional patterns

**Empirical Evidence:**

Research shows different heads learn different patterns:

**Head 1 (Syntactic):**
In "The cat sat on the mat":
- "cat" attends to "The" (determiner)
- "sat" attends to "cat" (subject-verb)
- Learns grammatical structure

**Head 2 (Semantic):**
- "cat" attends to "mat" (both physical objects)
- "sat" attends to "on" (action-preposition)
- Learns meaning relationships

**Head 3 (Positional):**
- Each token attends to adjacent tokens
- Learns local context
- Similar to n-grams

**Head 4 (Long-range):**
- Attends to distant relevant tokens
- "mat" at end attends to "cat" at beginning
- Captures document-level context

**Mathematical Benefit:**

d_model = 512, num_heads = 8
Each head: d_k = 512/8 = 64 dimensions

Advantages:
- Lower dimensional projections (64 vs 512)
- More efficient computation
- Specialization without interference
- Ensemble effect (multiple perspectives)

**Practical Impact:**

Studies show removing any single head decreases performance, but removing all heads except one causes severe degradation. The combination of specialized heads is powerful.`,
    keyPoints: [
      'Multiple heads learn specialized patterns: syntax, semantics, position',
      'Each head operates in lower-dimensional space (d_model/num_heads)',
      'Different heads capture different aspects simultaneously',
      'Ensemble effect: combining multiple perspectives improves accuracy',
      'Empirically validated: different heads show distinct attention patterns',
    ],
  },
  {
    id: 'attention-nlp-dq-3',
    question:
      'Compare attention mechanisms with traditional seq2seq with LSTMs. What fundamental limitations of LSTMs does attention solve?',
    sampleAnswer: `Attention solves critical limitations of LSTM-based seq2seq models:

**LSTM Seq2Seq Limitations:**1. **Fixed-size bottleneck:**
   - Entire source sentence compressed into single hidden state vector
   - Long sentences lose information
   - Fixed size regardless of source length

2. **Sequential processing:**
   - Must process tokens one-by-one
   - Cannot parallelize
   - Slow training

3. **Long-range dependency degradation:**
   - Information decays over many timesteps
   - Distant tokens poorly represented in final state

4. **No explicit alignment:**
   - Implicit learning of source-target correspondence
   - Cannot visualize what model focuses on

**How Attention Solves These:**

**1. Variable-size context:**
- Encoder produces sequence of states (not single vector)
- Decoder can access ALL encoder states
- No compression bottleneck
- Context adapts to source length

**2. Parallel computation:**
- All attention scores computed simultaneously
- Matrix operations (highly optimized)
- 10-100x faster training on GPUs

**3. Direct connections:**
- Any decoder position directly accesses any encoder position
- No information degradation
- O(1) path length vs O(n) in LSTMs

**4. Explicit alignment:**
- Attention weights show correspondence
- Interpretable: visualize what input decoder focuses on
- Helps debugging and understanding

**Concrete Example:**

Translating long sentence (50 words):

**LSTM Seq2Seq:**
- Encoder: word₁→h₁→word₂→h₂→...→word₅₀→h₅₀
- h₅₀ must remember entire sentence (bottleneck!)
- Decoder uses only h₅₀
- Information from word₁ heavily degraded

**Attention Seq2Seq:**
- Encoder: produces [h₁, h₂, ..., h₅₀]
- Decoder at step t: attends to ALL encoder states
- Can focus on word₁ directly when needed
- No information loss

**Performance Impact:**

BLEU scores (machine translation):
- LSTM Seq2Seq: ~25-30
- LSTM + Attention: ~32-35  
- Transformer (full attention): ~38-42

Attention provides ~5-7 BLEU point improvement, transformers another 5-7.

**Key Insight:**

Attention fundamentally changes architecture from "compress then decode" to "attend then decode". This eliminates the bottleneck and enables modern NLP breakthroughs.`,
    keyPoints: [
      'LSTM bottleneck: compresses entire sequence into single vector',
      'Attention: decoder accesses ALL encoder states dynamically',
      'Enables parallel computation vs sequential LSTM processing',
      'Direct connections: O(1) vs O(n) path length for dependencies',
      'Explicit alignment: interpretable attention weights',
      'Performance: ~5-7 BLEU point improvement in translation',
    ],
  },
];
