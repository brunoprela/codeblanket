/**
 * Multiple choice questions for Image-to-Video Animation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const imagetovideanimationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-i2v-mc-1',
    question:
      'What does the motion_bucket_id parameter in Stable Video Diffusion control?',
    options: [
      'The resolution of the output video',
      'The amount of motion/animation applied to the image',
      'The number of frames generated',
      'The frame rate of the output',
    ],
    correctAnswer: 1,
    explanation:
      'The motion_bucket_id (0-255) controls animation intensity. Lower values (0-40) create subtle motion, medium values (80-127) create moderate motion, and high values (180-255) create dramatic motion.',
  },
  {
    id: 'vag-i2v-mc-2',
    question:
      'What is the recommended motion_bucket_id range for animating portrait photos?',
    options: [
      '0-20 (minimal motion)',
      '20-60 (subtle motion)',
      '100-150 (active motion)',
      '180-255 (maximum motion)',
    ],
    correctAnswer: 1,
    explanation:
      'Portraits should use 20-60 for subtle motion like breathing or gentle head movement. Higher values can cause uncanny valley effects with distorted facial features.',
  },
  {
    id: 'vag-i2v-mc-3',
    question:
      'Why is image-to-video generally more cost-effective than text-to-video for product demonstrations?',
    options: [
      'It uses smaller models',
      'It generates shorter videos',
      'It starts with a guaranteed correct first frame and only animates',
      'It requires less GPU memory',
    ],
    correctAnswer: 2,
    explanation:
      'Image-to-video is 2-4x cheaper because it only needs to animate an existing image rather than generate everything from scratch. This also guarantees product accuracy.',
  },
  {
    id: 'vag-i2v-mc-4',
    question:
      'What is the primary challenge when creating seamless video loops with image-to-video?',
    options: [
      'The first and last frames must match perfectly',
      'The video must be exactly 60 frames long',
      'The motion must always be circular',
      'The image must be square',
    ],
    correctAnswer: 0,
    explanation:
      'For seamless loops, the last frame must visually match the first frame so the video can repeat without a jarring transition. This often requires playing the video forward then backward (ping-pong).',
  },
  {
    id: 'vag-i2v-mc-5',
    question:
      'Which quality metric is most important for evaluating temporal consistency in generated videos?',
    options: [
      'Resolution (pixels)',
      'File size (MB)',
      'SSIM (Structural Similarity) between adjacent frames',
      'Frame rate (FPS)',
    ],
    correctAnswer: 2,
    explanation:
      'SSIM between adjacent frames measures how consistent the video is over time. High SSIM (>0.85) indicates good temporal consistency without flickering or morphing.',
  },
];
