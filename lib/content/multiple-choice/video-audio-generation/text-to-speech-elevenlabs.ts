/**
 * Multiple choice questions for Text-to-Speech (ElevenLabs) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const texttospeechelevenlabsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-tts-mc-1',
    question: 'What does the stability slider in ElevenLabs API control?',
    options: [
      'Audio quality/bitrate',
      'Consistency vs. expressiveness of the voice',
      'Speech rate/tempo',
      'Volume level',
    ],
    correctAnswer: 1,
    explanation:
      'Stability (0.0-1.0) controls consistency. High values (0.7-1.0) produce stable, consistent reads. Low values (0.0-0.3) allow more expressive variation but may be unpredictable.',
  },
  {
    id: 'vag-tts-mc-2',
    question:
      'How much audio is required to create a high-quality voice clone with ElevenLabs?',
    options: ['10-30 seconds', '1-3 minutes', '5-15 minutes', '30-60 minutes'],
    correctAnswer: 1,
    explanation:
      'Professional Voice Cloning requires 1-3 minutes of clean, consistent audio. More than 5 minutes offers diminishing returns, while less than 1 minute produces lower quality.',
  },
  {
    id: 'vag-tts-mc-3',
    question:
      'What is the primary benefit of using ElevenLabs streaming endpoint?',
    options: [
      'Higher audio quality',
      'Lower cost per character',
      'Reduced time-to-first-audio and better UX',
      'Support for more languages',
    ],
    correctAnswer: 2,
    explanation:
      "Streaming returns audio chunks as they're generated, allowing playback to start in ~300-800ms instead of waiting for complete generation. This significantly improves perceived responsiveness.",
  },
  {
    id: 'vag-tts-mc-4',
    question:
      'Which voice_settings parameter controls the emphasis and energy in the speech?',
    options: ['stability', 'similarity_boost', 'style', 'speaker_boost'],
    correctAnswer: 2,
    explanation:
      'The style parameter (0.0-1.0) controls exaggeration and emphasis. Higher values (0.7-1.0) produce more energetic, expressive speech with emphasis on key words.',
  },
  {
    id: 'vag-tts-mc-5',
    question:
      'What is the recommended approach for generating very long audiobooks (10+ hours)?',
    options: [
      'Send the entire text in one API call',
      'Split by chapters, track consistency with previous_text parameter',
      'Use the fastest model to reduce costs',
      'Disable streaming to ensure quality',
    ],
    correctAnswer: 1,
    explanation:
      'Split long content into chapters/sections and use the previous_text parameter (1000 chars) to maintain voice consistency across chunks. This prevents context loss and pronunciation drift.',
  },
];
