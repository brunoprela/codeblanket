/**
 * Multiple choice questions for Text-to-Video Models section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const texttovideommodelsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-t2v-mc-1',
    question:
      'What is the primary advantage of Stable Video Diffusion (SVD) over commercial APIs like Runway Gen-2?',
    options: [
      'Higher quality output',
      'Faster generation speed',
      'Open-source and self-hostable with no per-generation costs',
      'Longer maximum video length',
    ],
    correctAnswer: 2,
    explanation:
      'SVD is open-source, allowing self-hosting without per-generation fees. While commercial APIs may offer convenience, SVD provides cost savings at scale and full control over the model.',
  },
  {
    id: 'vag-t2v-mc-2',
    question:
      'Which model is best suited for rapid prototyping and quick iteration when cost is less important than speed?',
    options: [
      'Runway Gen-2',
      'Pika Labs',
      'Stable Video Diffusion',
      'AnimateDiff',
    ],
    correctAnswer: 1,
    explanation:
      'Pika Labs generates videos in 1-3 minutes, making it fastest for iteration. While limited to 3 seconds, its speed makes it ideal for rapid prototyping and testing concepts.',
  },
  {
    id: 'vag-t2v-mc-3',
    question:
      'What is the estimated cost difference between generating a 15-second video with Runway Gen-2 versus self-hosted Stable Video Diffusion?',
    options: [
      'Runway is 2-3x more expensive',
      'Runway is 5-10x more expensive',
      'They cost approximately the same',
      'SVD is more expensive due to GPU costs',
    ],
    correctAnswer: 1,
    explanation:
      'Runway Gen-2 costs ~$0.50/second ($7.50 for 15s). Self-hosted SVD on an A10G costs ~$1-2/hour total, generating multiple videos. At scale, Runway is 5-10x more expensive.',
  },
  {
    id: 'vag-t2v-mc-4',
    question:
      'Which approach provides the most control over animation style and character consistency?',
    options: [
      'Runway Gen-2 with detailed prompts',
      'Pika Labs with motion controls',
      'AnimateDiff with custom LoRAs and SD models',
      'Stable Video Diffusion with high motion values',
    ],
    correctAnswer: 2,
    explanation:
      'AnimateDiff works with any Stable Diffusion model and LoRAs, allowing complete control over style, characters, and aesthetics while adding motion. This provides maximum customization.',
  },
  {
    id: 'vag-t2v-mc-5',
    question:
      'When would you choose image-to-video over text-to-video generation?',
    options: [
      'When you want completely novel creations',
      'When you need guaranteed consistency of the first frame',
      'When you want longer videos',
      'When you need faster generation',
    ],
    correctAnswer: 1,
    explanation:
      'Image-to-video guarantees exact control over the first frame, ensuring character/object consistency and allowing reuse of existing assets like product photos or portraits.',
  },
];
