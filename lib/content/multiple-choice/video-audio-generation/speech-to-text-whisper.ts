/**
 * Multiple choice questions for Speech-to-Text (Whisper) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const speechtotextwhisperMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-whisper-mc-1',
    question:
      'What is the main difference between Whisper\'s "transcribe" and "translate" endpoints?',
    options: [
      'Transcribe is faster than translate',
      'Transcribe outputs the original language, translate outputs English',
      'Transcribe requires timestamps, translate does not',
      'Translate is more accurate than transcribe',
    ],
    correctAnswer: 1,
    explanation:
      'The transcribe endpoint outputs text in the original language, while the translate endpoint always converts to English regardless of input language.',
  },
  {
    id: 'vag-whisper-mc-2',
    question: "What is the maximum audio file size for OpenAI's Whisper API?",
    options: ['10 MB', '25 MB', '50 MB', '100 MB'],
    correctAnswer: 1,
    explanation:
      "OpenAI's Whisper API has a 25 MB limit. Longer files must be chunked, typically using silence detection to find natural break points every 10-15 minutes.",
  },
  {
    id: 'vag-whisper-mc-3',
    question:
      'Which Whisper response format provides both timestamps and speaker diarization hints?',
    options: ['text', 'json', 'verbose_json', 'srt'],
    correctAnswer: 2,
    explanation:
      'The verbose_json format includes word-level timestamps, confidence scores, and segment information that can help with diarization analysis. The basic json format has less detail.',
  },
  {
    id: 'vag-whisper-mc-4',
    question:
      'What is the primary challenge when using Whisper for real-time transcription?',
    options: [
      'It requires high-end GPUs',
      "It doesn't support streaming and processes complete audio chunks",
      'It only works offline',
      'It has high per-minute costs',
    ],
    correctAnswer: 1,
    explanation:
      'Whisper is not designed for streaming and requires complete audio segments. For real-time use, you must buffer audio (e.g., 5-30 seconds), then transcribe chunks, introducing latency.',
  },
  {
    id: 'vag-whisper-mc-5',
    question:
      'Which technique improves Whisper accuracy for domain-specific vocabulary (e.g., medical terms)?',
    options: [
      'Using the prompt parameter with example text containing the vocabulary',
      'Increasing the temperature parameter',
      'Using translate instead of transcribe',
      'Requesting verbose_json format',
    ],
    correctAnswer: 0,
    explanation:
      'The prompt parameter (up to 224 tokens) provides context and example vocabulary, helping Whisper correctly recognize domain-specific terms and maintain consistent spelling.',
  },
];
