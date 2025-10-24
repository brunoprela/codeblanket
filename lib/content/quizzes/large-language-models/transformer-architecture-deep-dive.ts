export const transformerArchitectureQuiz = {
  title: 'Transformer Architecture Discussion',
  id: 'transformer-architecture-quiz',
  sectionId: 'transformer-architecture-deep-dive',
  questions: [
    {
      id: 1,
      question:
        'Explain why self-attention was such a breakthrough compared to RNNs and LSTMs. How does the attention mechanism enable parallelization and long-range dependencies? Discuss the computational complexity tradeoffs and why transformers have largely replaced RNNs despite the O(n²) complexity of attention.',
      expectedAnswer:
        'Should cover: sequential bottleneck in RNNs preventing parallelization, vanishing gradients in long sequences, direct connections in attention enabling long-range dependencies, parallel computation of all positions, O(n²) vs O(n) tradeoffs, practical performance despite theoretical complexity, role of GPUs in making attention efficient, and innovations like sparse attention to address quadratic scaling.',
    },
    {
      id: 2,
      question:
        'The original transformer paper introduced multi-head attention with 8 heads. Explain what each attention head learns and why having multiple heads is beneficial. How does the model learn to specialize different heads for different linguistic phenomena, and what happens if you reduce or increase the number of heads?',
      expectedAnswer:
        'Should discuss: attention heads as independent learned feature extractors, specialization to different aspects (syntax, semantics, coreference), ensemble benefits of multiple perspectives, empirical observations of head specialization, redundancy vs diversity tradeoffs, pruning studies showing some heads are unnecessary, optimal head count being task-dependent, relationship between head count and model width, and interpretability challenges in understanding head functions.',
    },
    {
      id: 3,
      question:
        "Analyze the role of positional encodings in transformers. Why are they necessary, and what are the tradeoffs between sinusoidal encodings, learned positional embeddings, and relative position encodings (like RoPE and ALiBi)? How do these choices affect the model's ability to generalize to longer sequences than seen during training?",
      expectedAnswer:
        "Should explain: permutation invariance of attention requiring position information, fixed vs learned encodings tradeoffs, sine/cosine encoding enabling extrapolation, relative vs absolute position benefits, RoPE's rotation- based approach and advantages, ALiBi's attention bias method, generalization to longer contexts, computational efficiency differences, and how positional encoding choice affects fine-tuning and deployment.",
    },
  ],
};
