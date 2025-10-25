/**
 * Multiple choice questions for Video Generation Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const videogenerationfundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'vag-vidgenfund-mc-1',
      question:
        'What is the primary reason video generation is significantly more challenging than image generation?',
      options: [
        'Videos require more storage space',
        'Maintaining temporal consistency across hundreds of frames',
        'Video models have more parameters',
        'Users have higher quality expectations for videos',
      ],
      correctAnswer: 1,
      explanation:
        'Temporal consistency is the fundamental challenge - objects must maintain identity, motion must be physically plausible, and style must remain coherent across all frames. A single inconsistent frame can break the entire video.',
    },
    {
      id: 'vag-vidgenfund-mc-2',
      question:
        "How does Sora\'s spacetime patch approach differ from earlier video generation methods?",
      options: [
        'It processes frames completely independently',
        'It treats time as another spatial dimension in the transformer',
        'It only uses text conditioning without image data',
        'It generates videos at lower resolution',
      ],
      correctAnswer: 1,
      explanation:
        'Sora treats video as spacetime patches where time is handled similarly to spatial dimensions, allowing the transformer to learn temporal relationships naturally through self-attention across both space and time simultaneously.',
    },
    {
      id: 'vag-vidgenfund-mc-3',
      question:
        'What is the approximate computational cost ratio for generating a 5-second video at 24fps compared to a single high-resolution image?',
      options: [
        '5-10x more expensive',
        '50-100x more expensive',
        '100-500x more expensive',
        '1000x or more expensive',
      ],
      correctAnswer: 2,
      explanation:
        'A 5-second video at 24fps is 120 frames. With temporal modeling overhead and the need for consistency checks, this is typically 100-500x more computationally expensive than generating a single image.',
    },
    {
      id: 'vag-vidgenfund-mc-4',
      question:
        'Which temporal consistency technique is most effective for preventing flickering in style-transferred videos?',
      options: [
        'Processing all frames independently and averaging results',
        'Using optical flow to warp previous frame results as conditioning',
        'Only processing keyframes and interpolating others',
        'Reducing the frame rate to minimize inconsistencies',
      ],
      correctAnswer: 1,
      explanation:
        "Optical flow-guided processing warps the previous frame's result using motion vectors and blends it with the current frame's processing, maintaining consistency by explicitly accounting for motion between frames.",
    },
    {
      id: 'vag-vidgenfund-mc-5',
      question:
        'What is the estimated training cost for a Sora-scale video generation model?',
      options: [
        '$1-5 million',
        '$10-25 million',
        '$50-100+ million',
        '$500 million+',
      ],
      correctAnswer: 2,
      explanation:
        'Training a Sora-scale model requires thousands of GPUs for months, processing millions of hours of video. The compute cost alone is estimated at $50-100+ million, not including data acquisition, storage, and engineering costs.',
    },
  ];
