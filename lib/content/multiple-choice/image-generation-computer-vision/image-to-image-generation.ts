import { MultipleChoiceQuestion } from '../../../types';

export const imagetoimagegenerationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-img2img-mc-1',
    question: 'What does the strength parameter of 0.5 mean in img2img?',
    options: [
      'Generate 50% of the time',
      '50% noise added before denoising',
      '50% quality',
      '50% speed',
    ],
    correctAnswer: 1,
    explanation:
      'Strength determines how much noise is added before denoising. 0.5 means moderate changes while preserving structure.',
  },
  {
    id: 'igcv-img2img-mc-2',
    question:
      'Which strength range is best for style transfer while preserving composition?',
    options: ['0.1-0.3', '0.4-0.6', '0.7-0.9', '1.0'],
    correctAnswer: 1,
    explanation:
      '0.4-0.6 provides balanced style transfer, changing appearance while preserving overall composition.',
  },
  {
    id: 'igcv-img2img-mc-3',
    question: 'What is the key advantage of img2img over txt2img?',
    options: [
      'Faster generation',
      'Better quality',
      'Composition control from input image',
      'Lower cost',
    ],
    correctAnswer: 2,
    explanation:
      'img2img uses the input image for structural guidance, giving better control over composition.',
  },
  {
    id: 'igcv-img2img-mc-4',
    question: 'For subtle color grading, what strength should you use?',
    options: ['0.2-0.3', '0.5-0.6', '0.7-0.8', '0.9-1.0'],
    correctAnswer: 0,
    explanation:
      'Low strength (0.2-0.3) makes subtle changes like color grading while preserving all details.',
  },
  {
    id: 'igcv-img2img-mc-5',
    question: 'Can you use img2img with any image as input?',
    options: [
      'Only photos',
      'Only AI-generated images',
      'Yes, any image',
      'Only same-size images as output',
    ],
    correctAnswer: 2,
    explanation:
      'img2img works with any image as input - photos, drawings, AI-generated, etc.',
  },
];
