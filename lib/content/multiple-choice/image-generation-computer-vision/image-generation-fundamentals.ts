import { MultipleChoiceQuestion } from '../../../types';

export const imagegenerationfundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'igcv-imgfund-mc-1',
      question:
        'What is the primary advantage of working in latent space rather than pixel space for image generation?',
      options: [
        'Better image quality',
        'Computational efficiency (~48x smaller)',
        'Easier to understand',
        'More accurate colors',
      ],
      correctAnswer: 1,
      explanation:
        'Latent space is ~48x smaller than pixel space, making generation much faster and requiring less GPU memory.',
    },
    {
      id: 'igcv-imgfund-mc-2',
      question: 'In diffusion models, what does the "forward process" do?',
      options: [
        'Removes noise from images',
        'Adds noise progressively to images',
        'Compresses images',
        'Generates new images',
      ],
      correctAnswer: 1,
      explanation:
        'The forward diffusion process progressively adds noise to images during training. The model learns to reverse this for generation.',
    },
    {
      id: 'igcv-imgfund-mc-3',
      question:
        'What is the typical range for the guidance_scale parameter in Stable Diffusion?',
      options: [
        '0.1 to 1.0',
        '1.0 to 5.0',
        '7.0 to 9.0 (recommended)',
        '20.0 to 30.0',
      ],
      correctAnswer: 2,
      explanation:
        'Guidance scale 7-9 is recommended, balancing prompt adherence with creative freedom.',
    },
    {
      id: 'igcv-imgfund-mc-4',
      question:
        'Which model would be most cost-effective for generating 10,000 images per day?',
      options: [
        'DALL-E 3',
        'Midjourney',
        'Stable Diffusion (local)',
        'All are equally cost-effective',
      ],
      correctAnswer: 2,
      explanation:
        'Stable Diffusion running locally has zero per-image cost after initial hardware investment.',
    },
    {
      id: 'igcv-imgfund-mc-5',
      question: 'What does CLIP do in image generation pipelines?',
      options: [
        'Compresses images',
        'Encodes text prompts to embeddings',
        'Removes backgrounds',
        'Upscales images',
      ],
      correctAnswer: 1,
      explanation:
        'CLIP encodes text prompts into embeddings that guide the image generation process.',
    },
  ];
