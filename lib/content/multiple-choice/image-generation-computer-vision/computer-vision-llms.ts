import { MultipleChoiceQuestion } from '../../../types';

export const computervisionllmsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-cvllm-mc-1',
    question: 'What is the "detail" parameter in GPT-4V?',
    options: [
      'Image resolution setting',
      'Controls cost and processing detail level',
      'Color depth',
      'Zoom level',
    ],
    correctAnswer: 1,
    explanation:
      'Detail level (low/high/auto) controls processing thoroughness, affecting both cost and analysis quality.',
  },
  {
    id: 'igcv-cvllm-mc-2',
    question: 'Can GPT-4V extract text from images (OCR)?',
    options: [
      'No',
      'Yes, very effectively',
      'Only from printed text',
      'Only from typed text',
    ],
    correctAnswer: 1,
    explanation:
      'GPT-4V can effectively extract and understand text from images, making it excellent for OCR tasks.',
  },
  {
    id: 'igcv-cvllm-mc-3',
    question: 'Which is more expensive for simple image classification?',
    options: [
      'detail=low',
      'detail=high',
      'Both cost the same',
      'Neither costs anything',
    ],
    correctAnswer: 1,
    explanation:
      'detail=high costs ~3x more than detail=low ($0.03 vs $0.01 per image), so use low for simple tasks.',
  },
  {
    id: 'igcv-cvllm-mc-4',
    question: 'Can vision LLMs compare multiple images in one request?',
    options: [
      'No, one at a time only',
      'Yes, multiple images can be included',
      'Only two images',
      'Only with special API',
    ],
    correctAnswer: 1,
    explanation:
      'Vision LLMs like GPT-4V can analyze and compare multiple images in a single request.',
  },
  {
    id: 'igcv-cvllm-mc-5',
    question:
      'What is a key advantage of vision LLMs over traditional computer vision?',
    options: [
      'Faster processing',
      'Understanding and reasoning, not just detection',
      'Lower cost',
      'Higher accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Vision LLMs can understand context and reason about images, not just detect objects.',
  },
];
