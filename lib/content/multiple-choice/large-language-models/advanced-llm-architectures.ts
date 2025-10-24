export const advancedLLMArchitecturesMC = {
  title: 'Advanced LLM Architectures Quiz',
  id: 'advanced-llm-architectures-mc',
  sectionId: 'advanced-llm-architectures',
  questions: [
    {
      id: 1,
      question:
        'In Mixture of Experts (MoE) models, what does "sparse activation" mean?',
      options: [
        'Only some neurons fire during training',
        'Only K out of N expert networks are activated for each token',
        'Activations are stored in sparse matrices',
        'The model is partially loaded into memory',
      ],
      correctAnswer: 1,
      explanation:
        'MoE uses sparse activation—for each token, a router selects K experts (e.g., 2 out of 8) to process it. Only those K experts compute, making inference cheaper than a dense model with equivalent parameters. Total params >> active params.',
    },
    {
      id: 2,
      question:
        'What is the estimated architecture of GPT-4 according to public speculation?',
      options: [
        'Single 1 trillion parameter dense model',
        'Mixture of Experts with ~8 experts of ~220B each',
        'Ensemble of 100 smaller models',
        'Standard transformer with no architectural innovations',
      ],
      correctAnswer: 1,
      explanation:
        'While unconfirmed, credible speculation suggests GPT-4 uses MoE with ~8 experts of ~220B parameters each (~1.8T total), with 2 experts active per token (~440B active). This explains its high capability with manageable inference costs.',
    },
    {
      id: 3,
      question:
        'What innovation does ALiBi (Attention with Linear Biases) introduce for handling longer contexts?',
      options: [
        'Compresses attention to linear complexity',
        'Adds linear bias to attention based on distance instead of positional embeddings',
        'Uses linear layers instead of attention',
        'Linearly reduces context window',
      ],
      correctAnswer: 1,
      explanation:
        'ALiBi replaces positional embeddings with biases added to attention scores based on token distance. This is simpler, saves parameters, and enables better extrapolation to sequences longer than trained on—models trained on 1k can handle 10k+ at inference.',
    },
    {
      id: 4,
      question: 'How do multimodal models like GPT-4V process images?',
      options: [
        'Convert images to text descriptions',
        'Use a separate vision encoder (Vision Transformer) to create visual tokens',
        'Train the text model directly on pixel values',
        'Download images and analyze them externally',
      ],
      correctAnswer: 1,
      explanation:
        'Multimodal models use a vision encoder (typically a Vision Transformer/ViT) to convert images into token embeddings, which are then processed alongside text tokens by the LLM. Cross-modal attention allows the model to integrate visual and textual information.',
    },
    {
      id: 5,
      question:
        'What is the primary challenge with long context models using sparse attention patterns?',
      options: [
        'Too much memory usage',
        'Reduced quality/recall compared to full attention',
        'Incompatible with modern GPUs',
        'Slower than full attention',
      ],
      correctAnswer: 1,
      explanation:
        'Sparse attention patterns (local windows, strided patterns) are approximations that sacrifice some information flow compared to full O(n²) attention. The challenge is maintaining quality while achieving efficiency—different patterns (Longformer, BigBird) try different tradeoffs.',
    },
  ],
};
