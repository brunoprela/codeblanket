import { QuizQuestion } from '../../../types';

export const sequenceModelingNlpQuiz: QuizQuestion[] = [
  {
    id: 'sequence-modeling-nlp-dq-1',
    question:
      'Explain the vanishing gradient problem in vanilla RNNs and how LSTMs solve it with their gating mechanism. Why is this critical for NLP tasks?',
    sampleAnswer: `The vanishing gradient problem prevents RNNs from learning long-term dependencies, which LSTMs solve through their cell state and gates.

**Vanishing Gradient Problem:**

In vanilla RNNs, gradients are backpropagated through time:
- At each timestep: gradient × tanh'(h_t)
- tanh derivative ≤ 0.25
- After 50 timesteps: 0.25^50 ≈ 10^-30 (vanishes!)

This makes it impossible to learn dependencies longer than ~10-20 words.

**Why Critical for NLP:**

Many NLP tasks require long-range dependencies:
- "The cat, which we found yesterday in the garden, was hungry" (subject-verb agreement across 10+ words)
- "Alice went to the bank" ... 20 words later ... "she withdrew money" (coreference)

Vanilla RNNs cannot learn these patterns.

**How LSTMs Solve This:**

LSTM cell state provides a "highway" for gradients:
- Cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
- Gradients flow through cell state with minimal modification
- Gates control information flow, not transformation
- Gradient doesn't need to backprop through many tanh operations

**Gates:**1. Forget gate (f): What to keep from previous cell state
2. Input gate (i): What new information to add
3. Output gate (o): What to output from cell

This allows gradients to flow efficiently across 100+ timesteps.`,
    keyPoints: [
      'Vanilla RNN gradients vanish due to repeated tanh derivatives',
      'Makes learning dependencies beyond ~10-20 words impossible',
      'LSTMs use cell state as gradient highway',
      'Gates control flow without transforming gradients',
      'Enables learning dependencies across 100+ words',
      'Critical for NLP: subject-verb agreement, coreference, long context',
    ],
  },
  {
    id: 'sequence-modeling-nlp-dq-2',
    question:
      'Compare bidirectional LSTMs with unidirectional LSTMs for NLP tasks. When would you use each, and what are the trade-offs?',
    sampleAnswer: `Bidirectional LSTMs process sequences in both directions, providing fuller context but with trade-offs.

**Unidirectional LSTM:**
- Processes left-to-right only
- At position t: sees words 0...t-1
- Cannot see future context

**Bidirectional LSTM:**
- Two LSTMs: forward (left-to-right) and backward (right-to-left)
- At position t: sees entire sentence
- Concatenates both directions: [h_forward; h_backward]

**When to Use Bidirectional:**

Use for understanding tasks:
1. **Text Classification**: "This movie was great!"
   - Needs full sentence to classify sentiment
   - Bidirectional sees "great" even when processing "movie"

2. **Named Entity Recognition**: "Apple announced new iPhone"
   - "Apple" could be fruit or company
   - Backward pass sees "announced", "iPhone" → company
   
3. **Question Answering**: Need full context of passage

**When to Use Unidirectional:**

Use for generation tasks:
1. **Language Modeling**: Predict next word
   - Can only see past, not future
   - Bidirectional would "cheat" by seeing answer

2. **Text Generation**: Generate word-by-word
   - Must generate sequentially
   - No future context exists yet

3. **Real-time Processing**: Process as text arrives
   - Cannot wait for full sentence

**Trade-offs:**

Bidirectional advantages:
- Better accuracy (+2-5% typically)
- Full context for each word
- Better for understanding

Bidirectional disadvantages:
- 2x parameters and compute
- Cannot generate sequences
- Requires full sequence upfront
- Not suitable for streaming

**Practical Example:**

Sentiment analysis: "The movie started slow but ended great"
- Unidirectional at "slow": negative signal, might misclassify
- Bidirectional: sees "but ended great" → correct positive classification`,
    keyPoints: [
      'Bidirectional processes both directions, unidirectional only left-to-right',
      'Bidirectional better for understanding: classification, NER, QA',
      'Unidirectional necessary for generation: cannot see future',
      'Bidirectional: 2x compute, cannot generate, needs full sequence',
      'Trade-off: accuracy vs generation capability vs compute',
    ],
  },
  {
    id: 'sequence-modeling-nlp-dq-3',
    question:
      'LSTMs were state-of-the-art for NLP but have largely been replaced by transformers. What limitations of LSTMs led to this shift, and when might LSTMs still be preferred today?',
    sampleAnswer: `LSTMs have fundamental limitations that transformers overcome, though LSTMs remain useful in specific scenarios.

**LSTM Limitations:**1. **Sequential Processing:**
   - Must process tokens one-by-one
   - Cannot parallelize across sequence
   - Token t depends on hidden state from token t-1
   - Training bottleneck: slow on modern GPUs

2. **Limited Context Window:**
   - Despite solving vanishing gradients, effective context ~100-200 tokens
   - Information still degrades over very long sequences
   - Hidden state is a bottleneck

3. **No Direct Long-Range Connections:**
   - Information flows through hidden states
   - Distant words connected through many intermediate steps
   - Harder to learn direct relationships

**How Transformers Improved:**1. **Parallelization:**
   - Self-attention processes all tokens simultaneously
   - Fully utilizes GPU parallelism
   - 10-100x faster training

2. **Unlimited Context:**
   - Every token attends directly to every other token
   - No distance decay
   - BERT: 512 tokens, GPT-3: 2048+, modern: 100K+

3. **Direct Connections:**
   - Any word directly attends to any other word
   - Explicit attention weights show relationships
   - Better interpretability

**When LSTMs Still Preferred:**1. **Resource-Constrained Environments:**
   - Smaller model size
   - Lower memory requirements
   - Edge devices, mobile

2. **Streaming/Online Processing:**
   - Process token-by-token as they arrive
   - Don't need full sequence
   - Real-time applications

3. **Sequential Generation:**
   - Natural for autoregressive generation
   - Simpler than transformer decoders
   - Text generation, speech synthesis

4. **Time Series Forecasting:**
   - Stock prices, sensor data
   - Sequential nature is natural
   - Often outperform transformers for pure time series

5. **Small Data Scenarios:**
   - Fewer parameters to train
   - Less prone to overfitting
   - When <10K training examples

6. **Interpretability:**
   - Hidden states more intuitive than attention
   - Easier to understand model decisions
   - Regulated industries

**Performance Comparison:**

Task benchmarks (approximate):
- Sentiment analysis: LSTM 85%, BERT 92%
- NER: LSTM 88%, BERT 95%
- Question answering: LSTM 75%, BERT 88%

Transformers consistently outperform by 5-10% absolute.

**Practical Recommendation:**

Modern projects: Start with transformers (BERT, GPT)
- Better accuracy
- Pre-trained models available
- Industry standard

Consider LSTMs when:
- Strict latency requirements (<10ms)
- Very limited memory (<100MB model)
- Streaming required
- Time series data
- Educational purposes

**Hybrid Approaches:**

Some applications use both:
- LSTM for sequential processing
- Transformer for attention
- Best of both worlds

**Historical Context:**

2015-2017: LSTMs dominated NLP
2018: Transformer revolution (BERT, GPT)
2024: Transformers standard, LSTMs niche

LSTMs remain important for understanding sequence modeling and still useful in specific scenarios.`,
    keyPoints: [
      'LSTM limitation: sequential processing, cannot parallelize',
      "Limited context window ~100-200 tokens vs transformer's thousands",
      'Transformers: parallel, unlimited context, direct connections',
      'LSTMs still useful: resource-constrained, streaming, small data',
      'Transformers 5-10% better accuracy but 10-100x more compute',
      'Modern standard: transformers; LSTMs for niche applications',
    ],
  },
];
