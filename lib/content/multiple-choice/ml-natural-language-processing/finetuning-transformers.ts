import { MultipleChoiceQuestion } from '../../../types';

export const finetuningTransformersMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'finetuning-transformers-mc-1',
    question:
      'What is the primary advantage of fine-tuning over training from scratch?',
    options: [
      'Fine-tuning is always faster',
      'Fine-tuning leverages pre-trained knowledge, requiring less data and achieving better performance',
      'Fine-tuning uses less memory',
      'Fine-tuning does not require GPUs',
    ],
    correctAnswer: 1,
    explanation:
      'Fine-tuning leverages knowledge from pre-training on massive corpora (Wikipedia, books, web text). This transfer learning enables strong performance with small datasets (even 1000s of examples vs millions needed from scratch) and often achieves better final accuracy.',
  },
  {
    id: 'finetuning-transformers-mc-2',
    question: 'What is LoRA (Low-Rank Adaptation)?',
    options: [
      'A type of transformer architecture',
      'A parameter-efficient fine-tuning method that trains <1% of parameters',
      'A new pre-training objective',
      'A tokenization method',
    ],
    correctAnswer: 1,
    explanation:
      'LoRA injects trainable low-rank decomposition matrices into transformer layers, training only these small matrices (~0.1-1% of total parameters) while keeping the base model frozen. This achieves 90%+ of full fine-tuning performance with 10-100x less compute and memory.',
  },
  {
    id: 'finetuning-transformers-mc-3',
    question:
      'What learning rate is typically recommended for fine-tuning BERT?',
    options: [
      '1e-3 (0.001)',
      '1e-1 (0.1)',
      '2e-5 (0.00002)',
      '1e-6 (0.000001)',
    ],
    correctAnswer: 2,
    explanation:
      'Fine-tuning transformers requires much smaller learning rates (2e-5 to 5e-5) than training from scratch. Higher learning rates can destabilize the pre-trained weights and cause loss spikes or NaN. This is 10-100x smaller than typical deep learning learning rates.',
  },
  {
    id: 'finetuning-transformers-mc-4',
    question: 'Why is learning rate warmup important in fine-tuning?',
    options: [
      'To save GPU memory',
      'To gradually increase LR from 0, preventing large updates that destabilize early training',
      'To speed up training',
      'To reduce overfitting',
    ],
    correctAnswer: 1,
    explanation:
      'Warmup gradually increases learning rate from 0 to target value over first 5-10% of steps. This prevents large gradient updates early in training that could destabilize the carefully pre-trained weights, especially important when fine-tuning from pre-trained checkpoints.',
  },
  {
    id: 'finetuning-transformers-mc-5',
    question:
      'When should you freeze the base transformer layers during fine-tuning?',
    options: [
      'Always freeze to save compute',
      'Never freeze for best accuracy',
      'Freeze when you have very small datasets (<1000 examples) to prevent overfitting',
      'Freeze only the embedding layer',
    ],
    correctAnswer: 2,
    explanation:
      'Freezing base layers is useful for very small datasets (<1000 examples) where full fine-tuning would overfit. The frozen layers act as a feature extractor, and only the task-specific head is trained. For larger datasets (>10K examples), full fine-tuning typically works better.',
  },
];
