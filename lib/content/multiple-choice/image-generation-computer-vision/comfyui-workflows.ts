import { MultipleChoiceQuestion } from '../../../types';

export const comfyuiworkflowsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'igcv-comfy-mc-1',
    question: 'What is ComfyUI?',
    options: [
      'A Stable Diffusion model',
      'A node-based workflow builder for SD',
      'A cloud service',
      'An upscaling tool',
    ],
    correctAnswer: 1,
    explanation:
      'ComfyUI is a powerful node-based visual workflow builder for Stable Diffusion.',
  },
  {
    id: 'igcv-comfy-mc-2',
    question: 'What is the main advantage of ComfyUI over simple UIs?',
    options: [
      'Faster generation',
      'Better quality',
      'Complex workflow control and reusability',
      'Lower cost',
    ],
    correctAnswer: 2,
    explanation:
      'ComfyUI allows building complex multi-step workflows that can be saved and reused.',
  },
  {
    id: 'igcv-comfy-mc-3',
    question: 'Can you use ComfyUI through an API?',
    options: [
      'No, UI only',
      'Yes, it has an API',
      'Only with paid version',
      'Only for simple operations',
    ],
    correctAnswer: 1,
    explanation:
      'ComfyUI provides an API for programmatic workflow execution and automation.',
  },
  {
    id: 'igcv-comfy-mc-4',
    question: 'What format are ComfyUI workflows saved as?',
    options: ['Binary', 'JSON', 'XML', 'Python'],
    correctAnswer: 1,
    explanation:
      'ComfyUI workflows are saved as JSON, making them easy to share and version control.',
  },
  {
    id: 'igcv-comfy-mc-5',
    question:
      'Can you combine multiple techniques (img2img, ControlNet, etc.) in one ComfyUI workflow?',
    options: [
      'No, one technique only',
      'Yes, that is a key benefit',
      'Only two techniques',
      'Only with plugins',
    ],
    correctAnswer: 1,
    explanation:
      'A key benefit of ComfyUI is combining multiple techniques in one workflow.',
  },
];
