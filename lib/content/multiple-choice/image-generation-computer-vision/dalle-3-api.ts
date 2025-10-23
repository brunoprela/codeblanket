import { MultipleChoiceQuestion } from '../../../types';

export const dalle3apiMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-dalle3-mc-1',
    question:
      'What is the cost difference between standard and HD quality in DALL-E 3 for a 1024×1024 image?',
    options: [
      '$0.02 vs $0.04',
      '$0.04 vs $0.08',
      '$0.08 vs $0.16',
      'Both cost the same',
    ],
    correctAnswer: 1,
    explanation:
      'DALL-E 3 costs $0.04 for standard and $0.08 for HD quality (2x cost) for square 1024×1024 images.',
  },
  {
    id: 'igcv-dalle3-mc-2',
    question: 'What does the "detail" parameter control in GPT-4V?',
    options: [
      'Image resolution',
      'Cost and processing detail level',
      'Color accuracy',
      'Generation speed',
    ],
    correctAnswer: 1,
    explanation:
      'The detail parameter (low/high/auto) controls how much detail the model processes, affecting both cost and quality.',
  },
  {
    id: 'igcv-dalle3-mc-3',
    question: 'How many images can DALL-E 3 generate per request?',
    options: ['1', '4', '10', 'Unlimited'],
    correctAnswer: 0,
    explanation:
      'DALL-E 3 can only generate 1 image per request (n=1 only), unlike earlier versions.',
  },
  {
    id: 'igcv-dalle3-mc-4',
    question: 'What is a "revised prompt" in DALL-E 3?',
    options: [
      'Your original prompt corrected for errors',
      'An enhanced version of your prompt created by DALL-E',
      'A prompt template',
      'A saved prompt',
    ],
    correctAnswer: 1,
    explanation:
      'DALL-E 3 automatically enhances your prompt with more details, returning this as revised_prompt.',
  },
  {
    id: 'igcv-dalle3-mc-5',
    question:
      'Which style produces more dramatic, cinematic images in DALL-E 3?',
    options: ['natural', 'vivid', 'realistic', 'standard'],
    correctAnswer: 1,
    explanation:
      'The "vivid" style produces hyper-real, dramatic, cinematic images, while "natural" is more realistic and subdued.',
  },
];
