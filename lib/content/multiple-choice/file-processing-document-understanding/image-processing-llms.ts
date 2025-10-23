/**
 * Multiple choice questions for Image Processing for LLMs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const imageprocessingllmsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-image-proc-mc-1',
    question: 'What is the purpose of image preprocessing before OCR?',
    options: [
      'To make the image look better',
      'To reduce file size',
      'To improve text recognition accuracy by enhancing contrast and removing noise',
      'To convert the image to color',
    ],
    correctAnswer: 2,
    explanation:
      'Preprocessing (grayscale conversion, contrast enhancement, denoising, binarization) improves OCR accuracy by making text clearer and reducing interference from backgrounds and noise.',
  },
  {
    id: 'fpdu-image-proc-mc-2',
    question: 'Which vision LLM API requires images in base64 format?',
    options: [
      'Only GPT-4V',
      'Only Claude 3',
      'Both require base64 for image data',
      'Neither, they accept image URLs only',
    ],
    correctAnswer: 2,
    explanation:
      'Both GPT-4V and Claude 3 accept base64-encoded images in their API. They also support image URLs, but for local files, base64 encoding is required.',
  },
  {
    id: 'fpdu-image-proc-mc-3',
    question: 'What library is commonly used for OCR in Python?',
    options: ['opencv-python', 'pytesseract', 'scikit-image', 'matplotlib'],
    correctAnswer: 1,
    explanation:
      'pytesseract is the Python wrapper for Tesseract OCR, the most widely used open-source OCR engine. It provides text extraction with position and confidence information.',
  },
  {
    id: 'fpdu-image-proc-mc-4',
    question: 'When analyzing a complex diagram, which approach is BEST?',
    options: [
      'Use only pytesseract OCR',
      'Use only vision LLM (GPT-4V or Claude 3)',
      'Manually trace the diagram elements',
      'Convert to text file first',
    ],
    correctAnswer: 1,
    explanation:
      'Vision LLMs like GPT-4V and Claude 3 excel at understanding diagrams, relationships, and visual structure. OCR alone cannot interpret visual relationships and diagram semantics.',
  },
  {
    id: 'fpdu-image-proc-mc-5',
    question: 'What is the recommended way to reduce vision LLM API costs?',
    options: [
      'Send images at maximum resolution for best quality',
      'Call the API multiple times to verify results',
      'Resize images to minimum required resolution and cache results',
      'Use the most expensive model for all requests',
    ],
    correctAnswer: 2,
    explanation:
      'Resizing images to the minimum required resolution reduces token usage and costs. Caching results by image hash prevents redundant API calls. Both strategies significantly reduce costs.',
  },
];
