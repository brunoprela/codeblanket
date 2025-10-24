export const llmFundamentalsMC = {
  title: 'LLM Fundamentals Quiz',
  id: 'llm-fundamentals-mc',
  sectionId: 'llm-fundamentals',
  questions: [
    {
      id: 1,
      question:
        'What is the primary training objective for GPT-style decoder-only language models?',
      options: [
        'Masked language modeling (predicting masked tokens bidirectionally)',
        'Next token prediction (autoregressive left-to-right)',
        'Sequence-to-sequence translation',
        'Contrastive learning between positive and negative pairs',
      ],
      correctAnswer: 1,
      explanation:
        "GPT models use causal/autoregressive language modeling, predicting the next token given all previous tokens. This is different from BERT's masked language modeling which uses bidirectional context.",
    },
    {
      id: 2,
      question:
        'What phenomenon describes capabilities that suddenly appear in large language models at certain scales but are absent in smaller models?',
      options: [
        'Transfer learning',
        'Overfitting',
        'Emergent abilities',
        'Catastrophic forgetting',
      ],
      correctAnswer: 2,
      explanation:
        'Emergent abilities are capabilities that appear at certain model scales (often >100B parameters) but are not present in smaller models. Examples include arithmetic, multi-step reasoning, and instruction following.',
    },
    {
      id: 3,
      question:
        'Which component is unique to encoder-only models like BERT compared to decoder-only models like GPT?',
      options: [
        'Self-attention mechanism',
        'Bidirectional attention across entire sequence',
        'Positional encodings',
        'Layer normalization',
      ],
      correctAnswer: 1,
      explanation:
        'BERT uses bidirectional attention, allowing each token to attend to all tokens (left and right). GPT uses causal attention where tokens can only attend to previous tokens. Other components exist in both architectures.',
    },
    {
      id: 4,
      question:
        'What is the typical ratio of training tokens to model parameters recommended by the Chinchilla scaling laws?',
      options: [
        '1:1 (equal tokens and parameters)',
        '10:1 (10 tokens per parameter)',
        '20:1 (20 tokens per parameter)',
        '100:1 (100 tokens per parameter)',
      ],
      correctAnswer: 2,
      explanation:
        'Chinchilla scaling laws suggest approximately 20 tokens per parameter for compute-optimal training. This means a 70B parameter model should be trained on ~1.4T tokens, contradicting earlier approaches that used fewer tokens.',
    },
    {
      id: 5,
      question:
        'What is the primary advantage of decoder-only architectures (like GPT) over encoder-decoder architectures (like T5) for general-purpose use?',
      options: [
        'Lower computational requirements',
        'Better performance on understanding tasks',
        'Unified architecture for both understanding and generation',
        'Faster inference speed',
      ],
      correctAnswer: 2,
      explanation:
        'Decoder-only models provide a unified architecture that can handle both understanding (via completion) and generation naturally. They also excel at few-shot learning through in-context examples, though encoder-decoder models can have advantages for specific seq2seq tasks.',
    },
  ],
};
