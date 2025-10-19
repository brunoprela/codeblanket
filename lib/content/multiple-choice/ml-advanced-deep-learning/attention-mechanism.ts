/**
 * Attention Mechanism Multiple Choice Questions
 */
export const attentionmechanismMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question:
      'What is the primary problem that attention mechanisms solve in Seq2Seq models?',
    options: [
      'They make training faster by parallelizing computations',
      'They solve the fixed-size context vector bottleneck by allowing the decoder to access all encoder hidden states',
      'They eliminate the need for LSTM/GRU cells',
      'They reduce the number of parameters in the model',
    ],
    correctAnswer: 1,
    explanation:
      "Attention solves the **bottleneck problem** where all input information must be compressed into a fixed-size context vector. Without attention: All encoder states → single vector c (512-dim) → decoder uses only c. Information loss for long sequences. With attention: All encoder states {h_1,...,h_n} preserved → decoder computes dynamic context c_t = Σ α_{t,i} × h_i at each step → can access any encoder position. Benefits: (1) No compression loss, (2) Performance doesn't degrade with length, (3) Direct access to relevant information. This breakthrough (2015) improved translation quality dramatically, especially for long sequences (40+ words: 15→30 BLEU score!). Attention doesn't primarily speed up training, eliminate RNNs, or reduce parameters - it fundamentally changes information flow.",
  },
  {
    id: 'cnn-mc-2',
    question:
      'In attention mechanism, what does the softmax operation over scores accomplish?',
    options: [
      'It speeds up the computation by removing negative values',
      'It normalizes scores into a probability distribution that sums to 1, creating interpretable attention weights',
      'It makes all scores equal so every encoder position is treated the same',
      'It selects only the single highest scoring position',
    ],
    correctAnswer: 1,
    explanation:
      "Softmax converts raw scores into **normalized probability distribution**: α_i = exp(e_i) / Σ_j exp(e_j). Properties: (1) All α_i ∈ [0, 1], (2) Σ α_i = 1 (valid distribution), (3) Differentiable (trainable end-to-end), (4) Amplifies differences (high scores → much higher probabilities). Example: Scores [2.0, 0.5, 0.1] → Softmax [0.73, 0.20, 0.07]. This creates interpretable weights we can visualize. Without softmax, couldn't combine encoder states meaningfully (raw scores unbounded). Softmax doesn't remove negatives (exp handles that), doesn't equalize (preserves relative magnitudes), and doesn't select single position (weighted sum of all). The '1-sum' property is crucial for weighted averaging.",
  },
  {
    id: 'cnn-mc-3',
    question:
      'What is the key difference between Bahdanau (additive) and Luong (multiplicative) attention?',
    options: [
      'Bahdanau uses previous decoder state s_{t-1} for scoring, while Luong uses current state s_t',
      'Bahdanau is only for translation, Luong works for all tasks',
      'Bahdanau requires more training data than Luong',
      'Luong attention cannot handle variable-length sequences',
    ],
    correctAnswer: 0,
    explanation:
      'The key difference is **timing** of when attention is computed: **Bahdanau**: (1) Compute attention using s_{t-1} (previous decoder state), (2) Get context c_t, (3) Update decoder: s_t = LSTM(s_{t-1}, [y_{t-1}; c_t]), (4) Output from s_t. Attention BEFORE decoder update. **Luong**: (1) Update decoder: s_t = LSTM(s_{t-1}, y_{t-1}), (2) Compute attention using s_t (current state), (3) Get context c_t, (4) Output from [s_t; c_t]. Attention AFTER decoder update. Score functions also differ: Bahdanau uses additive (v^T tanh(W₁h_i + W₂s)), Luong uses multiplicative (h_i^T W s). Performance similar (~23-24 BLEU), but Luong ~30% faster. Both work for all sequence tasks, handle variable lengths, and need similar data. The timing difference is subtle but fundamental to their design.',
  },
  {
    id: 'cnn-mc-4',
    question:
      'When visualizing attention weights as a heatmap for translation, what does a strong diagonal pattern indicate?',
    options: [
      'The model is making errors and needs more training',
      'The source and target languages have similar word order, with roughly word-by-word alignment',
      'The attention mechanism is not working properly',
      'The sequences have exactly the same length',
    ],
    correctAnswer: 1,
    explanation:
      "A **diagonal pattern** indicates monotonic alignment where source word i corresponds to target word i: Position 1→1, 2→2, 3→3, etc. This means **similar word order** between languages. Example: English 'The cat sat' → French 'Le chat était' shows diagonal (word-by-word). This is GOOD - model learned sensible alignment! Not an error. Other patterns are also valid: (1) Horizontal bands: One source → many targets ('not' → 'ne...pas'), (2) Vertical bands: Many sources → one target ('will go' → 'ira'), (3) Reverse diagonal: Major reordering (SOV ↔ SVO languages). Diagonal doesn't require equal length - attention is per-position, not overall shape. A diffuse/scattered pattern (no clear structure) indicates problems, but clear patterns of any type show model learned meaningful alignments.",
  },
  {
    id: 'cnn-mc-5',
    question:
      'In attention mechanism, the context vector c_t is computed as a weighted sum: c_t = Σ α_{t,i} × h_i. Why is this formulation better than just using the encoder hidden state with highest attention weight?',
    options: [
      'Using a weighted sum is computationally faster than selecting a single state',
      'Weighted sum allows the model to incorporate information from multiple relevant positions simultaneously, and is differentiable',
      'Selecting the maximum would require sorting, which is too slow',
      'Weighted sum always uses all encoder states equally',
    ],
    correctAnswer: 1,
    explanation:
      "**Weighted sum** is superior to \"argmax\" (selecting highest) because: **1. Multiple Relevant Positions**: Often multiple source words are relevant. Example: Translating 'machine learning' → 'apprentissage automatique' needs information from BOTH words. Weighted sum: c_t = 0.6×h_'machine' + 0.4×h_'learning' incorporates both! Argmax: Would pick only 'machine', lose 'learning' info. **2. Differentiability**: Weighted sum is differentiable w.r.t. attention weights: ∂c_t/∂α_{t,i} = h_i. Can backpropagate and train end-to-end! Argmax is non-differentiable (discrete selection). **3. Soft Decision**: Attention weights encode uncertainty. If unsure between positions, can hedge bets (0.5, 0.5). Argmax forces hard choice. **4. Gradual Learning**: During training, model can gradually shift attention rather than switching abruptly. NOT about speed (weighted sum slower than argmax), and doesn't use all equally (weights can be 0.9, 0.05, 0.03, 0.02...). Weighted sum is fundamental to attention's success - combines information smoothly and trains end-to-end.",
  },
];
