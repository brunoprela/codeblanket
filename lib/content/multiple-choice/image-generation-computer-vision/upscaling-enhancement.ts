import { MultipleChoiceQuestion } from '../../../types';

export const upscalingenhancementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-upscale-mc-1',
    question:
      'What is the main advantage of AI upscaling over traditional methods?',
    options: [
      'Faster processing',
      'Generates realistic details vs blurring',
      'Lower cost',
      'Smaller file sizes',
    ],
    correctAnswer: 1,
    explanation:
      'AI upscaling generates realistic details that should be there, while traditional methods just blur/interpolate.',
  },
  {
    id: 'igcv-upscale-mc-2',
    question: 'Which tool is the industry standard for AI upscaling?',
    options: ['Photoshop', 'Real-ESRGAN', 'DALL-E', 'Stable Diffusion'],
    correctAnswer: 1,
    explanation:
      'Real-ESRGAN is the go-to tool for fast, high-quality AI upscaling.',
  },
  {
    id: 'igcv-upscale-mc-3',
    question: 'For anime/illustration upscaling, which model should you use?',
    options: [
      'RealESRGAN_x4plus',
      'RealESRGAN_x4plus_anime_6B',
      'SD Upscale',
      'DALL-E Upscale',
    ],
    correctAnswer: 1,
    explanation:
      'RealESRGAN_x4plus_anime_6B is specialized for anime and illustrations, preserving line art quality.',
  },
  {
    id: 'igcv-upscale-mc-4',
    question: 'What is progressive upscaling?',
    options: [
      'Showing progress bar',
      'Upscaling in multiple smaller steps',
      'Upscaling parts of image',
      'Faster upscaling',
    ],
    correctAnswer: 1,
    explanation:
      'Progressive upscaling applies multiple smaller scale steps (e.g., 2x→2x→2x for 8x) for better quality.',
  },
  {
    id: 'igcv-upscale-mc-5',
    question: 'How much can Real-ESRGAN typically upscale images?',
    options: ['2x only', '4x typically', '10x', 'Unlimited'],
    correctAnswer: 1,
    explanation:
      'Real-ESRGAN typically provides 4x upscaling (e.g., 512×512 to 2048×2048).',
  },
];
