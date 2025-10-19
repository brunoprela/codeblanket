import { MultipleChoiceQuestion } from '../../../types';

export const transformerModelsNlpMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'transformer-models-nlp-mc-1',
    question: 'What is the key difference between BERT and GPT architectures?',
    options: [
      'BERT is larger than GPT',
      'BERT uses bidirectional attention (encoder), GPT uses unidirectional (decoder)',
      'BERT is faster than GPT',
      'BERT requires more training data',
    ],
    correctAnswer: 1,
    explanation:
      'BERT uses only the encoder with bidirectional self-attention, allowing it to see context from both directions. GPT uses only the decoder with causal (unidirectional) attention, seeing only previous tokens. This makes BERT better for understanding tasks and GPT better for generation.',
  },
  {
    id: 'transformer-models-nlp-mc-2',
    question: 'Why are positional encodings necessary in transformer models?',
    options: [
      'To reduce computational complexity',
      'To handle variable-length sequences',
      'Because attention is permutation-invariant and word order information would be lost',
      'To improve training speed',
    ],
    correctAnswer: 2,
    explanation:
      'Unlike RNNs which process sequentially, attention mechanisms treat all positions equally and are permutation-invariant. Without positional encodings, "dog bites man" would be identical to "man bites dog". Positional encodings explicitly add position information to preserve word order.',
  },
  {
    id: 'transformer-models-nlp-mc-3',
    question: 'What pre-training objective does BERT use?',
    options: [
      'Next token prediction',
      'Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)',
      'Text generation',
      'Translation',
    ],
    correctAnswer: 1,
    explanation:
      'BERT uses Masked Language Modeling (randomly masking 15% of tokens and predicting them) and Next Sentence Prediction (predicting if sentence B follows sentence A). This bidirectional pre-training enables BERT to learn rich contextual representations.',
  },
  {
    id: 'transformer-models-nlp-mc-4',
    question:
      'What is the computational complexity of self-attention in transformers?',
    options: [
      'O(n) where n is sequence length',
      'O(n log n)',
      'O(n²) where n is sequence length',
      'O(1)',
    ],
    correctAnswer: 2,
    explanation:
      'Self-attention has O(n²) complexity because each of n positions attends to all n positions, requiring n² attention score computations. This quadratic complexity is why transformers struggle with very long sequences and why efficient variants (Linformer, Longformer) have been developed.',
  },
  {
    id: 'transformer-models-nlp-mc-5',
    question: 'What is DistilBERT?',
    options: [
      'A variant of BERT that adds more layers',
      'A compressed version of BERT via knowledge distillation, 40% smaller and 60% faster',
      'BERT trained on more data',
      'BERT with different tokenization',
    ],
    correctAnswer: 1,
    explanation:
      "DistilBERT is created through knowledge distillation, training a smaller student model to mimic BERT (the teacher). It achieves 40% size reduction, 60% speedup, while retaining 97% of BERT's performance, making it ideal for production where speed and size matter.",
  },
];
