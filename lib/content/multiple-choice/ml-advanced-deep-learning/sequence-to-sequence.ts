/**
 * Sequence-to-Sequence Models Multiple Choice Questions
 */
export const sequenceToSequenceMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question: 'In a Seq2Seq model, what role does the context vector play?',
    options: [
      'It stores the vocabulary for both source and target languages',
      "It\'s the fixed-size representation of the entire input sequence that the decoder uses to generate output",
      'It determines the learning rate during training',
      "It\'s used only during inference, not training",
    ],
    correctAnswer: 1,
    explanation:
      "The context vector is the **compressed representation** of the entire input sequence. Process: (1) Encoder reads input sequence token by token, updating hidden state, (2) Final encoder hidden state = context vector (typically 512-1024 dimensions), (3) This vector is passed to decoder as initial state, (4) Decoder generates output based on this context. Example: Translating 'I love cats' → Encoder compresses to context vector [0.2, -0.5, ...] (512 dims) → Decoder expands to 'J'aime les chats'. This is the **bottleneck**: all input information must fit in fixed size, limiting performance on long sequences. This bottleneck problem led to the invention of attention mechanisms.",
  },
  {
    id: 'cnn-mc-2',
    question: "What is 'teacher forcing' in Seq2Seq training?",
    options: [
      'A technique to force the model to learn faster by increasing batch size',
      "Using the true previous output token as input at each decoder step during training, rather than the model's prediction",
      'Forcing the encoder to compress the sequence into a smaller context vector',
      'A method to prevent overfitting by randomly dropping decoder connections',
    ],
    correctAnswer: 1,
    explanation:
      "Teacher forcing feeds the **true previous output** (not model's prediction) as the next input during training. Example: Generating 'The cat sat': Step 1: Input START → Predict 'The', Step 2: Input 'The' (TRUE, even if model predicted 'A') → Predict 'cat', Step 3: Input 'cat' (TRUE) → Predict 'sat'. Benefits: (1) Faster convergence - each step learns from correct context, (2) Stable training - no error accumulation, (3) Parallel computation possible. Problem: Exposure bias - model never sees its own errors during training, struggles at inference when using its predictions. Solution: Scheduled sampling - gradually reduce teacher forcing ratio from 1.0 to 0.2 over training.",
  },
  {
    id: 'cnn-mc-3',
    question:
      'Why does performance of basic Seq2Seq models degrade significantly on sequences longer than 30-40 tokens?',
    options: [
      'The LSTM cells run out of memory after 40 timesteps',
      "The fixed-size context vector can't adequately represent long sequences, causing information loss",
      'Teacher forcing stops working beyond 40 tokens',
      'The softmax function becomes numerically unstable',
    ],
    correctAnswer: 1,
    explanation:
      "The **fixed-size context vector bottleneck** causes degradation on long sequences. Problem: Whether input is 5 tokens or 500 tokens, context vector is same size (e.g., 512 dimensions). For 5 tokens: ~100 dims/token (adequate). For 100 tokens: ~5 dims/token (severe compression!). Consequences: (1) Information from early tokens gets 'overwritten' by later tokens, (2) Fine details lost in compression, (3) Model can't 'remember' everything in fixed space. Empirical evidence: Translation BLEU score: 10 words → 35 BLEU, 40 words → 25 BLEU (30% drop!). This is NOT an LSTM problem (LSTM solves vanishing gradients) - it's a capacity/bottleneck problem. Solution: Attention mechanism allows decoder to access all encoder states directly, no bottleneck.",
  },
  {
    id: 'cnn-mc-4',
    question:
      'In beam search decoding with beam width k=5, how many hypotheses does the model maintain at each step?',
    options: [
      'Only 1 (the best so far)',
      'Exactly 5 complete sequences',
      'Up to 5 partial sequences (beams), each representing a different hypothesis',
      '5 × vocabulary_size hypotheses',
    ],
    correctAnswer: 2,
    explanation:
      "Beam search maintains **up to k partial sequences** at each step. Process: (1) Start with 1 sequence [START], (2) Generate top-k tokens → 5 sequences, (3) For each sequence, generate top-k next tokens → 5×k=25 candidates, (4) Keep only top-5 of these 25, (5) Repeat until all beams end with <END>. Example with k=3: Step 1: ['The',], ['A',], ['This',]. Step 2: Expand each, keep top-3 overall: ['The','cat',], ['The','dog',], ['A','cat',]. Each beam is a partial sequence exploring different paths. Benefits: (1) Better than greedy (k=1) which picks single best token, (2) Explores multiple hypotheses, (3) Can recover if early token suboptimal. Trade-off: k× slower than greedy. Typical: k=5-10 gives good quality/speed balance.",
  },
  {
    id: 'cnn-mc-5',
    question:
      "When using a bidirectional encoder in Seq2Seq, why can't the decoder also be bidirectional?",
    options: [
      'Bidirectional LSTMs are too slow for decoding',
      'The decoder generates output autoregressively (one token at a time), so it can only look at previously generated tokens, not future ones',
      'Bidirectional decoders require twice as many parameters',
      "Teacher forcing doesn't work with bidirectional decoders",
    ],
    correctAnswer: 1,
    explanation:
      "The decoder must be **autoregressive** (unidirectional) because it generates output sequentially: Step 1: Generate y_1 based on [context]. Step 2: Generate y_2 based on [context, y_1]. Step 3: Generate y_3 based on [context, y_1, y_2]. At Step 2, y_3 doesn't exist yet! Can't look forward to something not generated. This is fundamental to sequential generation - we're producing output token-by-token, not analyzing a complete sequence. Contrast with encoder: Input sequence is complete, so we can process forward AND backward: forward pass sees [x_1...x_i], backward pass sees [x_n...x_i], concatenate for full context. Encoder=bidirectional ✓ (analyzing complete input), Decoder=unidirectional only (generating one-by-one).",
  },
];
