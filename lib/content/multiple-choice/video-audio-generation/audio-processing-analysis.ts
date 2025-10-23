/**
 * Multiple choice questions for Audio Processing & Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const audioprocessinganalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-audio-mc-1',
    question:
      'What is the primary reason to normalize audio to -1.0 to 1.0 range before processing with AI models?',
    options: [
      'It reduces file size',
      'It improves playback compatibility',
      'It ensures consistent model input scale and prevents numerical instability',
      'It increases audio quality',
    ],
    correctAnswer: 2,
    explanation:
      'AI models expect normalized input (-1.0 to 1.0 or 0.0 to 1.0). Unnormalized audio with varying amplitude ranges causes inconsistent predictions and potential numerical overflow/underflow.',
  },
  {
    id: 'vag-audio-mc-2',
    question:
      'Which sample rate is most commonly expected by speech-to-text models like Whisper?',
    options: ['8000 Hz', '16000 Hz', '44100 Hz', '48000 Hz'],
    correctAnswer: 1,
    explanation:
      'Most speech models (Whisper, Wav2Vec2) are trained on 16000 Hz audio. While higher sample rates work, resampling to 16kHz is standard and expected for optimal performance.',
  },
  {
    id: 'vag-audio-mc-3',
    question: 'What does spectral gating noise reduction primarily target?',
    options: [
      'Loud sudden noises (clicks, pops)',
      'Consistent background noise (hum, hiss, fan)',
      'Silence gaps',
      'Echo and reverb',
    ],
    correctAnswer: 1,
    explanation:
      'Spectral gating analyzes the frequency spectrum and attenuates consistent background noise across frequency bands. It learns the noise profile and subtracts it from the signal.',
  },
  {
    id: 'vag-audio-mc-4',
    question:
      'Why is Voice Activity Detection (VAD) important when chunking long audio files for Whisper?',
    options: [
      'It reduces file size',
      'It finds natural silence points to split audio without cutting mid-word',
      'It improves audio quality',
      'It detects multiple speakers',
    ],
    correctAnswer: 1,
    explanation:
      'VAD identifies speech vs. silence, allowing intelligent chunking at natural pauses. This prevents splitting in the middle of words/sentences, improving transcription accuracy.',
  },
  {
    id: 'vag-audio-mc-5',
    question:
      'What is the computational complexity advantage of using librosa over scipy for audio feature extraction?',
    options: [
      'librosa is always faster than scipy',
      'librosa has no advantage; scipy is faster',
      'librosa is optimized for audio and provides high-level functions, but scipy is faster for basic operations',
      'They have identical performance',
    ],
    correctAnswer: 2,
    explanation:
      'librosa provides convenient high-level audio functions (MFCC, chroma, spectrograms) but uses scipy/numpy under the hood. For basic operations like FFT, using scipy directly is faster.',
  },
];
