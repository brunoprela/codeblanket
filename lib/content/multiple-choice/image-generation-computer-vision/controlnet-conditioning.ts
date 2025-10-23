import { MultipleChoiceQuestion } from '../../../types';

export const controlnetconditioningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-cn-mc-1',
    question: 'What type of control does ControlNet-Canny provide?',
    options: [
      'Depth information',
      'Edge/outline control',
      'Color control',
      'Pose control',
    ],
    correctAnswer: 1,
    explanation:
      'ControlNet-Canny uses edge detection to control the outlines and structure of generated images.',
  },
  {
    id: 'igcv-cn-mc-2',
    question:
      'Which ControlNet type is best for maintaining exact human poses?',
    options: ['Canny', 'Depth', 'OpenPose', 'Scribble'],
    correctAnswer: 2,
    explanation:
      'OpenPose ControlNet detects and maintains exact human skeleton poses for precise character positioning.',
  },
  {
    id: 'igcv-cn-mc-3',
    question: 'What does the conditioning_scale parameter control?',
    options: [
      'Image size',
      'How strongly to follow the control image',
      'Generation speed',
      'Color saturation',
    ],
    correctAnswer: 1,
    explanation:
      'conditioning_scale (0-2.0) controls how strictly the generation follows the control input.',
  },
  {
    id: 'igcv-cn-mc-4',
    question:
      'Which ControlNet is best for architectural images with straight lines?',
    options: ['OpenPose', 'Scribble', 'MLSD', 'Depth'],
    correctAnswer: 2,
    explanation:
      'MLSD (M-LSD) detects and preserves straight lines, perfect for architecture and geometric structures.',
  },
  {
    id: 'igcv-cn-mc-5',
    question: 'Do you need to preprocess images for ControlNet?',
    options: ['Never', 'Yes, always', 'Only for some types', 'Only for photos'],
    correctAnswer: 1,
    explanation:
      'Most ControlNet types require preprocessing (e.g., edge detection for Canny, pose detection for OpenPose).',
  },
];
