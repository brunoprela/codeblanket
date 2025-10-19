/**
 * Transformer Architecture Multiple Choice Questions
 */

export const transformerArchitectureMultipleChoice = [
  {
    id: 'mc-1',
    question:
      'What is the main advantage of Transformers over RNNs for processing sequences?',
    options: [
      'Transformers use less memory than RNNs',
      'Transformers can process all positions in parallel, while RNNs must process sequentially',
      'Transformers have fewer parameters than RNNs',
      'Transformers work better on short sequences',
    ],
    correctAnswer: 1,
    explanation:
      "The primary advantage is **parallelization**: **RNNs (Sequential)**: h_1 → h_2 → h_3 → ... → h_n. Each step depends on previous, can't parallelize. Training time: O(n) sequential operations. **Transformers (Parallel)**: All positions processed simultaneously using self-attention. No sequential dependency during encoding. Training time: O(1) parallel operations (though O(n²) complexity per operation). Impact: (1) Training speed: 10-100× faster on modern GPUs, (2) Scalability: Can train on much larger datasets, (3) Long sequences: No vanishing gradients over distance. Example: For 512-token sequence: RNN needs 512 sequential steps, Transformer processes all 512 in one parallel operation. Memory usage is actually HIGHER for Transformers (O(n²) attention matrix), and parameter count is similar. The key win is parallelization enabling faster training and better GPU utilization.",
  },
  {
    id: 'mc-2',
    question:
      'In the Transformer, why do we scale the dot-product attention scores by 1/√d_k before applying softmax?',
    options: [
      'To make the computation faster',
      'To prevent the dot products from growing too large, which would push softmax into regions with small gradients',
      'To normalize the attention weights to sum to 1',
      'To reduce the memory requirements',
    ],
    correctAnswer: 1,
    explanation:
      "Scaling prevents **softmax saturation**: **Without scaling**: For large d_k, dot products Q·K grow in magnitude. Example with d_k=64: Two random vectors have E[Q·K] ≈ 0, but Var[Q·K] = d_k = 64. So scores might be [-15, 8, -20, 12]. **Softmax problem**: softmax(12) ≈ 1.0, softmax(-20) ≈ 0.0. Softmax saturates! Gradients: ∂softmax/∂score ≈ 0 for extreme values. Training stalls. **With scaling** (÷√64 = ÷8): Scores become [-1.9, 1.0, -2.5, 1.5]. Softmax: [0.02, 0.29, 0.01, 0.48]. Less extreme, gradients flow! **Why √d_k?** Variance of dot product scales with d_k. Dividing by √d_k makes variance ≈ 1 regardless of dimension. Scaling doesn't affect speed or memory (cheap division). Softmax itself normalizes to sum=1, that's separate. The scaling is specifically about keeping scores in a reasonable range for stable gradients.",
  },
  {
    id: 'mc-3',
    question:
      'Why do Transformers need positional encoding, while RNNs do not?',
    options: [
      'Transformers have more parameters and need the extra information',
      'Self-attention is permutation-invariant, so position information must be explicitly added',
      'Positional encoding makes training faster',
      'RNNs use positional encoding implicitly in their hidden states',
    ],
    correctAnswer: 1,
    explanation:
      "Self-attention is **permutation-invariant** - order doesn't matter: **Attention computation**: For each position i, compute α_{i,j} = softmax(q_i · k_j / √d_k), then output_i = Σ α_{i,j} v_j. Notice: This operates on SET of {v_j}, not SEQUENCE! If we permute input [A, B, C] → [C, A, B], the attention outputs would also permute identically. **Problem**: 'The cat sat' and 'sat cat The' would produce identical representations! **Solution**: Add position information to input embeddings. x_i' = embedding(token_i) + PE(position_i). Now position is baked into the representations. **RNNs don't need this**: h_t = f(h_{t-1}, x_t). Position information is IMPLICIT in the recurrence structure. h_3 inherently knows it came after h_2, which came after h_1. Sequential processing = built-in position awareness. Positional encoding doesn't speed up training or add useful parameters - it's NECESSARY for Transformers to understand order at all.",
  },
  {
    id: 'mc-4',
    question:
      'In multi-head attention with 8 heads and d_model=512, what is the dimension d_k of each head?',
    options: [
      'd_k = 512 (each head sees full dimension)',
      'd_k = 64 (d_model divided by num_heads)',
      'd_k = 8 (equal to number of heads)',
      'd_k = 4096 (d_model times num_heads)',
    ],
    correctAnswer: 1,
    explanation:
      "Each head uses **d_k = d_model / num_heads**: **Dimension splitting**: Total model dimension d_model = 512 must be distributed across h = 8 heads. Each head gets: d_k = d_v = 512 / 8 = 64. **Why split?** (1) **Computational efficiency**: Total computation stays same: 1 head × 512-dim ≈ 8 heads × 64-dim (both ≈ d_model²), (2) **Multiple perspectives**: Each 64-dim head learns different patterns, (3) **Parameter efficiency**: Same total parameters as single head. **Architecture**: Input x: (batch, seq_len, 512) → Linear projection → Reshape to (batch, seq_len, 8, 64) → Transpose to (batch, 8, seq_len, 64) → Attention per head → Combine back to (batch, seq_len, 512). **Not 512 per head**: That would be 8× more parameters and computation! **Not 8 or 4096**: Those don't match the math. The key insight: Split dimensions to enable multiple perspectives WITHOUT increasing computation.",
  },
  {
    id: 'mc-5',
    question:
      'The Transformer decoder uses masked self-attention. What does the mask prevent?',
    options: [
      'It prevents attending to padding tokens',
      'It prevents attending to future positions during training (causal masking)',
      'It prevents attending to the encoder outputs',
      'It prevents overfitting by randomly masking attention weights',
    ],
    correctAnswer: 1,
    explanation:
      "Causal masking prevents **seeing the future** during training: **Problem**: During training, we have full target sequence [y_1, y_2, y_3, y_4]. Without masking, when predicting y_3, self-attention could attend to y_4 (future). Model would cheat by copying the next token! **Solution**: Causal mask prevents position i from attending to positions j > i. Mask matrix (True = can attend): [[T, F, F, F], [T, T, F, F], [T, T, T, F], [T, T, T, T]]. Position 0: sees only position 0, Position 2: sees positions 0, 1, 2 (not 3). **Implementation**: Set masked scores to -∞ before softmax: scores[i, j] = -1e9 if j > i. After softmax, attention weight α_{i,j} ≈ 0 for j > i. **Autoregressive generation**: At inference, naturally causal (only have y_1,...,y_{t-1} when generating y_t). **Not padding mask** (that's separate), **not encoder masking** (that's cross-attention), **not dropout** (that's random, not structural). Causal masking enforces temporal dependency structure essential for sequence generation.",
  },
];
