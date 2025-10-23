/**
 * Multiple choice questions for Video Editing with AI section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const videoeditingwithaiMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vag-videdit-mc-1',
    question:
      'What is the main cause of flickering when applying style transfer frame-by-frame to videos?',
    options: [
      'Insufficient GPU memory',
      'Processing frames independently without temporal consistency',
      'Low video resolution',
      'Incorrect color space conversion',
    ],
    correctAnswer: 1,
    explanation:
      'Independent frame processing causes slight variations in style application between frames, creating visible flickering. Temporal consistency mechanisms that reference adjacent frames are needed.',
  },
  {
    id: 'vag-videdit-mc-2',
    question:
      'Which technique is most effective for maintaining temporal consistency in video style transfer?',
    options: [
      'Increasing frame rate',
      'Optical flow-guided warping of previous frame results',
      'Processing every other frame',
      'Applying stronger style weight',
    ],
    correctAnswer: 1,
    explanation:
      "Optical flow warps the previous frame's styled result to align with the current frame, then blends it with the current frame's processing. This maintains consistency by explicitly accounting for motion.",
  },
  {
    id: 'vag-videdit-mc-3',
    question:
      'What is the approximate computational cost multiplier for upscaling a video from 480p to 4K?',
    options: [
      '2x (double the pixels)',
      '4x (four times the pixels)',
      '8-10x (accounting for both resolution and temporal processing)',
      '16x (purely pixel count based)',
    ],
    correctAnswer: 2,
    explanation:
      '4K has 8.3x more pixels than 480p, but with temporal consistency requirements and tile-based processing overhead, the actual computational cost is 8-10x.',
  },
  {
    id: 'vag-videdit-mc-4',
    question:
      'When should you use optical flow methods over deep learning for frame interpolation?',
    options: [
      'Always, optical flow is always better',
      'Never, deep learning is always superior',
      'For small motion and high frame rate conversion (30fps to 60fps)',
      'Only for black and white videos',
    ],
    correctAnswer: 2,
    explanation:
      'Optical flow works well and is fast for small, smooth motion. Deep learning (RIFE, FILM) is better for complex motion, occlusions, and extreme slow-motion, but is much slower.',
  },
  {
    id: 'vag-videdit-mc-5',
    question:
      'What is the primary benefit of tile-based processing for video upscaling?',
    options: [
      'Faster processing speed',
      'Better quality results',
      'Fitting large videos in limited GPU memory',
      'Easier parallel processing across multiple GPUs',
    ],
    correctAnswer: 2,
    explanation:
      "Tile-based processing divides each frame into smaller tiles (e.g., 512x512) that fit in GPU memory, allowing upscaling of very high-resolution videos that wouldn't fit as complete frames.",
  },
];
