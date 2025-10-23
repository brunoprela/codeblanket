import { MultipleChoiceQuestion } from '../../../types';

export const stablediffusionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-sd-mc-1',
    question: 'How much VRAM is recommended for running SDXL?',
    options: ['4GB', '6GB', '8-12GB', '24GB'],
    correctAnswer: 2,
    explanation:
      "SDXL requires 8-12GB VRAM for comfortable operation, more than SD 2.1's 4-6GB requirement.",
  },
  {
    id: 'igcv-sd-mc-2',
    question: 'Which scheduler is fastest while maintaining good quality?',
    options: ['DDIM', 'DPM++ 2M', 'Euler Ancestral', 'LMS'],
    correctAnswer: 2,
    explanation:
      'Euler Ancestral (euler_a) offers the best balance of speed and quality, working well at 20-30 steps.',
  },
  {
    id: 'igcv-sd-mc-3',
    question: 'What does enable_attention_slicing() do?',
    options: [
      'Speeds up generation',
      'Reduces VRAM usage',
      'Improves quality',
      'Enables batch processing',
    ],
    correctAnswer: 1,
    explanation:
      'Attention slicing reduces VRAM usage by processing attention in smaller chunks, with minimal speed impact.',
  },
  {
    id: 'igcv-sd-mc-4',
    question: 'What is the native resolution of Stable Diffusion 2.1?',
    options: ['256×256', '512×512', '768×768', '1024×1024'],
    correctAnswer: 1,
    explanation:
      'SD 2.1 is trained on 512×512 images, producing best results at this resolution.',
  },
  {
    id: 'igcv-sd-mc-5',
    question:
      'Which optimization provides the biggest speed improvement for Stable Diffusion?',
    options: ['Attention slicing', 'VAE slicing', 'xformers', 'CPU offload'],
    correctAnswer: 2,
    explanation:
      'xformers provides the biggest speed improvement (20-40% faster) while also reducing memory usage.',
  },
];
