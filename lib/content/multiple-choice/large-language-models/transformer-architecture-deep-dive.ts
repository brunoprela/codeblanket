export const transformerArchitectureMC = {
  title: 'Transformer Architecture Quiz',
  id: 'transformer-architecture-mc',
  sectionId: 'transformer-architecture-deep-dive',
  questions: [
    {
      id: 1,
      question:
        'What is the computational complexity of standard self-attention with respect to sequence length n?',
      options: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
      correctAnswer: 2,
      explanation:
        'Self-attention has O(n²) complexity because each of the n tokens must attend to all n tokens, requiring n² pairwise computations. This becomes a bottleneck for very long sequences, leading to innovations like sparse attention.',
    },
    {
      id: 2,
      question:
        'In multi-head attention with 8 heads and model dimension 768, what is the dimension of each attention head?',
      options: [
        '768 (same as model dimension)',
        '384 (half the model dimension)',
        '96 (768 / 8)',
        '64 (fixed head dimension)',
      ],
      correctAnswer: 2,
      explanation:
        'Each head typically has dimension d_model / num_heads. With 768-dimensional model and 8 heads, each head is 96-dimensional. The outputs are concatenated back to 768 dimensions.',
    },
    {
      id: 3,
      question:
        'What is the purpose of the feedforward network (FFN) in each transformer layer?',
      options: [
        'Compute attention weights between tokens',
        'Apply non-linear transformations to token representations',
        'Generate positional encodings',
        'Normalize activations across the batch',
      ],
      correctAnswer: 1,
      explanation:
        'The FFN applies position-wise (independent per token) non-linear transformations, typically with an expansion ratio (4x wider hidden dimension). It processes each token independently after attention has mixed information across tokens.',
    },
    {
      id: 4,
      question:
        'Why did the original Transformer paper use sinusoidal positional encodings rather than learned embeddings?',
      options: [
        'Sinusoidal encodings train faster',
        'Sinusoidal encodings use less memory',
        'Sinusoidal encodings may generalize to longer sequences than seen during training',
        'Sinusoidal encodings are more interpretable',
      ],
      correctAnswer: 2,
      explanation:
        'Sinusoidal positional encodings have a deterministic pattern that theoretically allows the model to extrapolate to longer sequences than it was trained on. Learned embeddings are fixed to training length. In practice, both approaches work well.',
    },
    {
      id: 5,
      question:
        'In the attention mechanism, what are the three matrices learned for each attention head?',
      options: [
        'Input, Hidden, Output',
        'Query, Key, Value',
        'Encoder, Decoder, Output',
        'Position, Content, Attention',
      ],
      correctAnswer: 1,
      explanation:
        'Each attention head learns Query (Q), Key (K), and Value (V) projection matrices. The attention weights are computed from Q·Kᵀ, then used to weight the Values: Attention(Q,K,V) = softmax(QKᵀ/√d)V.',
    },
  ],
};
