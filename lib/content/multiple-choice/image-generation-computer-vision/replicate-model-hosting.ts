import { MultipleChoiceQuestion } from '../../../types';

export const replicatemodelhostingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-rep-mc-1',
    question: 'What is Replicate?',
    options: [
      'An AI model',
      'A cloud platform for running AI models',
      'A training service',
      'An upscaling tool',
    ],
    correctAnswer: 1,
    explanation:
      'Replicate is a cloud platform for running thousands of AI models via API.',
  },
  {
    id: 'igcv-rep-mc-2',
    question: 'How does Replicate pricing typically work?',
    options: [
      'Monthly subscription',
      'Pay per image',
      'Pay per second of compute',
      'Free with ads',
    ],
    correctAnswer: 2,
    explanation:
      'Replicate charges per second of compute time, making costs transparent and usage-based.',
  },
  {
    id: 'igcv-rep-mc-3',
    question: 'When is cloud hosting more cost-effective than local GPUs?',
    options: [
      'Always',
      'Never',
      'Low to moderate volume (<1000/day)',
      'High volume only',
    ],
    correctAnswer: 2,
    explanation:
      'Cloud is cost-effective for low-moderate volume; local becomes cheaper at high volume (1000s/day).',
  },
  {
    id: 'igcv-rep-mc-4',
    question: 'Does Replicate support custom trained models?',
    options: ['No', 'Yes', 'Only for enterprise', 'Only pre-trained models'],
    correctAnswer: 1,
    explanation:
      'Replicate supports deploying custom trained models, not just pre-trained ones.',
  },
  {
    id: 'igcv-rep-mc-5',
    question: 'What is a key advantage of cloud hosting?',
    options: [
      'Always cheapest',
      'No GPU required',
      'Better quality',
      'Faster than local',
    ],
    correctAnswer: 1,
    explanation:
      'Cloud hosting requires no local GPU hardware, enabling any device to generate images.',
  },
];
