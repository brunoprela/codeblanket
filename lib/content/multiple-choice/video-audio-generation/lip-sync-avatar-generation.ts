/**
 * Multiple choice questions for Lip Sync & Avatar Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const lipsyncavatargenerationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-lipsync-mc-1',
    question:
      'What is the primary input requirement for Wav2Lip to generate lip-synced videos?',
    options: [
      'Only audio file',
      'Only video file',
      'A video of a face and an audio file',
      'A 3D face model',
    ],
    correctAnswer: 2,
    explanation:
      'Wav2Lip requires a video containing a face (source video) and an audio file. It modifies the mouth movements in the video to match the audio speech.',
  },
  {
    id: 'vag-lipsync-mc-2',
    question:
      'What is the main advantage of D-ID and HeyGen APIs over open-source solutions like Wav2Lip?',
    options: [
      'They are free to use',
      'They provide better video quality, natural expressions, and head movements',
      'They work offline',
      'They support more languages',
    ],
    correctAnswer: 1,
    explanation:
      'Commercial APIs like D-ID and HeyGen produce more realistic results with natural expressions, head movements, and eye contact. Open-source solutions focus mainly on lip sync accuracy.',
  },
  {
    id: 'vag-lipsync-mc-3',
    question:
      'What is the typical resolution requirement for source images in D-ID avatar generation?',
    options: [
      '256x256 pixels minimum',
      '512x512 pixels minimum',
      'At least 256x256, but 512x512+ recommended for quality',
      '1920x1080 required',
    ],
    correctAnswer: 2,
    explanation:
      'D-ID accepts images as small as 256x256 but recommends 512x512 or higher for best quality. The face should be clearly visible and well-lit.',
  },
  {
    id: 'vag-lipsync-mc-4',
    question: 'What is a "presenter" in HeyGen API terminology?',
    options: [
      'The person recording the audio',
      'A pre-built or custom avatar that can deliver scripts',
      'The video editor',
      'A speech synthesis model',
    ],
    correctAnswer: 1,
    explanation:
      "A presenter is HeyGen's term for an avatar - either from their library or custom-created from user photos. Presenters can be assigned scripts to generate videos.",
  },
  {
    id: 'vag-lipsync-mc-5',
    question:
      'When should you use quality="high" in D-ID API vs. quality="low"?',
    options: [
      'Always use high for best results',
      'Use high for final production, low for rapid testing/previews',
      'Use low for better performance',
      'Quality setting has no impact',
    ],
    correctAnswer: 1,
    explanation:
      'quality="high" takes longer (30-90 seconds) but produces production-ready results. quality="low" is faster (10-30 seconds) and sufficient for testing scripts and timing.',
  },
];
