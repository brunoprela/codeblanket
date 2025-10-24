export const gptFamilyMC = {
  title: 'GPT Family Quiz',
  id: 'gpt-family-mc',
  sectionId: 'gpt-family',
  questions: [
    {
      id: 1,
      question:
        'What was the key innovation demonstrated by GPT-3 that was less prominent in GPT-2?',
      options: [
        'The transformer architecture',
        'Few-shot in-context learning at scale',
        'Instruction following through RLHF',
        'Multimodal capabilities',
      ],
      correctAnswer: 1,
      explanation:
        'GPT-3 demonstrated that with sufficient scale (175B parameters), models could perform few-shot learning by providing examples in the prompt, without any gradient updates. This was a qualitative leap from GPT-2.',
    },
    {
      id: 2,
      question:
        'Approximately how much more expensive per token is GPT-4 compared to GPT-3.5-Turbo for input tokens?',
      options: [
        '2x more expensive',
        '5x more expensive',
        '10-20x more expensive',
        '50x more expensive',
      ],
      correctAnswer: 2,
      explanation:
        'GPT-4 costs approximately $10-30/M tokens while GPT-3.5-Turbo costs $0.50-1.50/M tokens, making GPT-4 roughly 10-20x more expensive. This significant cost difference influences model selection in production.',
    },
    {
      id: 3,
      question: 'What is the context window size of GPT-4 Turbo?',
      options: ['8k tokens', '32k tokens', '128k tokens', '200k tokens'],
      correctAnswer: 2,
      explanation:
        "GPT-4 Turbo supports 128k tokens context window (roughly 300 pages), a significant increase from the original GPT-4's 8k. This enables processing of much longer documents.",
    },
    {
      id: 4,
      question:
        'What training technique, introduced between GPT-3 and GPT-4, helps align model outputs with human preferences?',
      options: [
        'Supervised fine-tuning only',
        'Reinforcement Learning from Human Feedback (RLHF)',
        'Adversarial training',
        'Curriculum learning',
      ],
      correctAnswer: 1,
      explanation:
        "RLHF (Reinforcement Learning from Human Feedback) was key to ChatGPT and GPT-4's success. It trains a reward model from human preferences, then optimizes the LLM using RL to maximize this reward while staying close to the original model.",
    },
    {
      id: 5,
      question: 'Why are GPT models called "autoregressive"?',
      options: [
        'They predict each token based on previous tokens sequentially',
        'They use automatic differentiation during training',
        'They regress to mean predictions',
        'They automatically tune hyperparameters',
      ],
      correctAnswer: 0,
      explanation:
        'Autoregressive means generating one token at a time, with each new token conditioned on all previously generated tokens. This is in contrast to non-autoregressive models that generate all tokens simultaneously.',
    },
  ],
};
