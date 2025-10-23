import { MultipleChoiceQuestion } from '../../../types';

export const advancedpromptingimagesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-advprompt-mc-1',
    question:
      'In prompt weighting syntax (keyword:1.5), what does the number represent?',
    options: [
      'Percentage',
      'Multiplier for emphasis',
      'Number of repetitions',
      'Priority ranking',
    ],
    correctAnswer: 1,
    explanation:
      '(keyword:1.5) means the keyword receives 1.5x normal emphasis/attention from the model.',
  },
  {
    id: 'igcv-advprompt-mc-2',
    question: 'What should be the first element in a well-structured prompt?',
    options: ['Style', 'Quality terms', 'Subject', 'Camera settings'],
    correctAnswer: 2,
    explanation:
      'Subject should come first in prompts as it establishes the primary focus and receives more attention.',
  },
  {
    id: 'igcv-advprompt-mc-3',
    question:
      'Which negative prompt term prevents anatomical errors in portraits?',
    options: ['blurry', 'bad anatomy', 'low quality', 'oversaturated'],
    correctAnswer: 1,
    explanation:
      'bad anatomy in negative prompts helps prevent common anatomical errors like extra fingers or distorted features.',
  },
  {
    id: 'igcv-advprompt-mc-4',
    question:
      'What is the recommended weight range for emphasis that is noticeable but not overwhelming?',
    options: ['0.5-0.8', '1.1-1.4', '1.5-2.0', '2.0+'],
    correctAnswer: 1,
    explanation:
      'Weights of 1.1-1.4 provide noticeable emphasis without overwhelming other elements or causing artifacts.',
  },
  {
    id: 'igcv-advprompt-mc-5',
    question: 'Which quality booster is most universally applicable?',
    options: [
      'trending on artstation',
      '8k',
      'professional photography',
      'highly detailed',
    ],
    correctAnswer: 3,
    explanation:
      'highly detailed is applicable across all styles and consistently improves output quality.',
  },
];
