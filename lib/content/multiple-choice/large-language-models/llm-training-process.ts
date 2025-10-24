export const llmTrainingProcessMC = {
  title: 'LLM Training Process Quiz',
  id: 'llm-training-process-mc',
  sectionId: 'llm-training-process',
  questions: [
    {
      id: 1,
      question:
        'What distributed training technique splits the model layers across multiple GPUs?',
      options: [
        'Data parallelism',
        'Tensor parallelism',
        'Pipeline parallelism',
        'Sequence parallelism',
      ],
      correctAnswer: 2,
      explanation:
        'Pipeline parallelism splits the model layers across GPUs (e.g., layers 1-10 on GPU0, layers 11-20 on GPU1). This enables training models too large for a single GPU, though it introduces pipeline bubbles that reduce efficiency.',
    },
    {
      id: 2,
      question:
        'What is the primary benefit of mixed precision training using FP16 or BF16?',
      options: [
        'Improved model accuracy',
        'Reduced memory usage and faster computation',
        'Easier debugging',
        'Better gradient flow',
      ],
      correctAnswer: 1,
      explanation:
        'Mixed precision training uses 16-bit floats (FP16/BF16) instead of 32-bit (FP32), reducing memory usage by ~2x and accelerating computation on modern GPUs. Loss scaling prevents numerical issues with small gradients.',
    },
    {
      id: 3,
      question:
        'According to the Chinchilla scaling laws, how should you allocate compute between model size and training data?',
      options: [
        'Maximize model size, data size is less important',
        'Maximize data size, model size is less important',
        'Balance model size and data tokens roughly equally (20:1 ratio)',
        'Use the minimum viable model size',
      ],
      correctAnswer: 2,
      explanation:
        'Chinchilla found that most models were undertrained. For compute-optimal training, you should train with approximately 20 tokens per parameter. This often means using a smaller model trained on more data than previous approaches.',
    },
    {
      id: 4,
      question:
        'What is gradient checkpointing used for in training large language models?',
      options: [
        'Saving model weights periodically',
        'Trading compute for memory by recomputing activations during backward pass',
        'Preventing gradient explosion',
        'Ensuring reproducible training runs',
      ],
      correctAnswer: 1,
      explanation:
        'Gradient checkpointing reduces memory usage by not storing all intermediate activations during forward pass. Instead, it recomputes them during backward pass as needed. This trades ~33% more compute for ~50% less memory.',
    },
    {
      id: 5,
      question:
        'What is the typical training dataset size for large language models like GPT-3 or LLaMA?',
      options: [
        '10-100 million tokens',
        '1-10 billion tokens',
        '100 billion - 2 trillion tokens',
        '10-100 trillion tokens',
      ],
      correctAnswer: 2,
      explanation:
        'Modern LLMs are trained on hundreds of billions to trillions of tokens. GPT-3 used ~300B tokens, while LLaMA-2 used ~2T tokens. Following Chinchilla scaling laws, a 70B parameter model should see ~1.4T tokens.',
    },
  ],
};
