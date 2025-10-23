import { MultipleChoiceQuestion } from '../../../types';

export const inpaintingeditingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-inpaint-mc-1',
    question: 'In an inpainting mask, what does white represent?',
    options: [
      'Areas to keep unchanged',
      'Areas to inpaint/change',
      'Transparent areas',
      'Highlighted areas',
    ],
    correctAnswer: 1,
    explanation:
      'White areas in the mask indicate regions to inpaint/change, black areas are preserved.',
  },
  {
    id: 'igcv-inpaint-mc-2',
    question: 'What is mask feathering/blur used for?',
    options: [
      'Making mask creation easier',
      'Smooth blending at edges',
      'Faster generation',
      'Better colors',
    ],
    correctAnswer: 1,
    explanation:
      'Feathering creates soft edges on the mask for seamless blending between inpainted and original areas.',
  },
  {
    id: 'igcv-inpaint-mc-3',
    question: 'What is outpainting?',
    options: [
      'Painting outside lines',
      'Extending image beyond original borders',
      'Removing objects',
      'Adding objects',
    ],
    correctAnswer: 1,
    explanation:
      'Outpainting extends images beyond their original borders, generating new content that continues the scene.',
  },
  {
    id: 'igcv-inpaint-mc-4',
    question: 'For object removal, what should the prompt describe?',
    options: [
      'The object to remove',
      'The background that should remain',
      'Both object and background',
      'Nothing specific',
    ],
    correctAnswer: 1,
    explanation:
      'When removing objects, prompt should describe what should fill the space (background continuation).',
  },
  {
    id: 'igcv-inpaint-mc-5',
    question: 'Why use specialized inpainting models instead of regular SD?',
    options: [
      'Faster generation',
      'Better seamless blending',
      'Higher resolution',
      'Lower cost',
    ],
    correctAnswer: 1,
    explanation:
      'Inpainting-specific models are trained to blend inpainted regions seamlessly with existing content.',
  },
];
