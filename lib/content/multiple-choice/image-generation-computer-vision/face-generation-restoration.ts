import { MultipleChoiceQuestion } from '../../../types';

export const facegenerationrestorationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'igcv-face-mc-1',
      question: 'What does GFPGAN do?',
      options: [
        'Generate faces',
        'Restore and enhance face quality',
        'Detect faces',
        'Remove faces',
      ],
      correctAnswer: 1,
      explanation:
        'GFPGAN (Generative Facial Prior GAN) restores and enhances degraded or AI-generated faces.',
    },
    {
      id: 'igcv-face-mc-2',
      question: 'Which model is better for severely damaged faces?',
      options: ['GFPGAN v1.3', 'GFPGAN v1.4', 'CodeFormer', 'Real-ESRGAN'],
      correctAnswer: 2,
      explanation:
        'CodeFormer often outperforms GFPGAN for severely damaged or low-quality faces.',
    },
    {
      id: 'igcv-face-mc-3',
      question: 'What does the weight parameter in face restoration control?',
      options: [
        'Output size',
        'Blend between original and restored',
        'Processing speed',
        'Color saturation',
      ],
      correctAnswer: 1,
      explanation:
        'Weight (0-1) controls blending: 0=original, 1=fully restored, 0.5=balanced mix.',
    },
    {
      id: 'igcv-face-mc-4',
      question: 'Why are faces particularly challenging for AI generation?',
      options: [
        'Too simple',
        'We are highly sensitive to facial errors',
        'Require special hardware',
        'Take too long',
      ],
      correctAnswer: 1,
      explanation:
        'Humans are extremely sensitive to facial errors (uncanny valley), making face generation challenging.',
    },
    {
      id: 'igcv-face-mc-5',
      question: 'What is face alignment used for?',
      options: [
        'Rotating images',
        'Standardizing face position for consistent processing',
        'Color correction',
        'Background removal',
      ],
      correctAnswer: 1,
      explanation:
        'Face alignment standardizes face position and orientation for consistent processing across images.',
    },
  ];
