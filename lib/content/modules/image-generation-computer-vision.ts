/**
 * Image Generation & Computer Vision Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { imagegenerationfundamentalsSection } from '../sections/image-generation-computer-vision/image-generation-fundamentals';
import { dalle3apiSection } from '../sections/image-generation-computer-vision/dalle-3-api';
import { stablediffusionSection } from '../sections/image-generation-computer-vision/stable-diffusion';
import { advancedpromptingimagesSection } from '../sections/image-generation-computer-vision/advanced-prompting-images';
import { imagetoimagegenerationSection } from '../sections/image-generation-computer-vision/image-to-image-generation';
import { controlnetconditioningSection } from '../sections/image-generation-computer-vision/controlnet-conditioning';
import { inpaintingeditingSection } from '../sections/image-generation-computer-vision/inpainting-editing';
import { facegenerationrestorationSection } from '../sections/image-generation-computer-vision/face-generation-restoration';
import { upscalingenhancementSection } from '../sections/image-generation-computer-vision/upscaling-enhancement';
import { computervisionllmsSection } from '../sections/image-generation-computer-vision/computer-vision-llms';
import { comfyuiworkflowsSection } from '../sections/image-generation-computer-vision/comfyui-workflows';
import { replicatemodelhostingSection } from '../sections/image-generation-computer-vision/replicate-model-hosting';
import { buildingimmagegenerationplatformSection } from '../sections/image-generation-computer-vision/building-image-generation-platform';

// Import quizzes
import { imagegenerationfundamentalsQuiz } from '../quizzes/image-generation-computer-vision/image-generation-fundamentals';
import { dalle3apiQuiz } from '../quizzes/image-generation-computer-vision/dalle-3-api';
import { stablediffusionQuiz } from '../quizzes/image-generation-computer-vision/stable-diffusion';
import { advancedpromptingimagesQuiz } from '../quizzes/image-generation-computer-vision/advanced-prompting-images';
import { imagetoimagegenerationQuiz } from '../quizzes/image-generation-computer-vision/image-to-image-generation';
import { controlnetconditioningQuiz } from '../quizzes/image-generation-computer-vision/controlnet-conditioning';
import { inpaintingeditingQuiz } from '../quizzes/image-generation-computer-vision/inpainting-editing';
import { facegenerationrestorationQuiz } from '../quizzes/image-generation-computer-vision/face-generation-restoration';
import { upscalingenhancementQuiz } from '../quizzes/image-generation-computer-vision/upscaling-enhancement';
import { computervisionllmsQuiz } from '../quizzes/image-generation-computer-vision/computer-vision-llms';
import { comfyuiworkflowsQuiz } from '../quizzes/image-generation-computer-vision/comfyui-workflows';
import { replicatemodelhostingQuiz } from '../quizzes/image-generation-computer-vision/replicate-model-hosting';
import { buildingimagegenerationplatformQuiz } from '../quizzes/image-generation-computer-vision/building-image-generation-platform';

// Import multiple choice
import { imagegenerationfundamentalsMultipleChoice } from '../multiple-choice/image-generation-computer-vision/image-generation-fundamentals';
import { dalle3apiMultipleChoice } from '../multiple-choice/image-generation-computer-vision/dalle-3-api';
import { stablediffusionMultipleChoice } from '../multiple-choice/image-generation-computer-vision/stable-diffusion';
import { advancedpromptingimagesMultipleChoice } from '../multiple-choice/image-generation-computer-vision/advanced-prompting-images';
import { imagetoimagegenerationMultipleChoice } from '../multiple-choice/image-generation-computer-vision/image-to-image-generation';
import { controlnetconditioningMultipleChoice } from '../multiple-choice/image-generation-computer-vision/controlnet-conditioning';
import { inpaintingeditingMultipleChoice } from '../multiple-choice/image-generation-computer-vision/inpainting-editing';
import { facegenerationrestorationMultipleChoice } from '../multiple-choice/image-generation-computer-vision/face-generation-restoration';
import { upscalingenhancementMultipleChoice } from '../multiple-choice/image-generation-computer-vision/upscaling-enhancement';
import { computervisionllmsMultipleChoice } from '../multiple-choice/image-generation-computer-vision/computer-vision-llms';
import { comfyuiworkflowsMultipleChoice } from '../multiple-choice/image-generation-computer-vision/comfyui-workflows';
import { replicatemodelhostingMultipleChoice } from '../multiple-choice/image-generation-computer-vision/replicate-model-hosting';
import { buildingimmagegenerationplatformMultipleChoice } from '../multiple-choice/image-generation-computer-vision/building-image-generation-platform';

export const imageGenerationComputerVisionModule: Module = {
  id: 'applied-ai-image-generation',
  title: 'Image Generation & Computer Vision',
  description:
    'Master AI-powered image generation and computer vision: from understanding diffusion models to building production image generation platforms with DALL-E, Stable Diffusion, and vision LLMs.',
  category: 'Applied AI',
  difficulty: 'Intermediate',
  estimatedTime: '18 hours',
  prerequisites: [
    'LLM Engineering Fundamentals',
    'Python proficiency',
    'Basic understanding of neural networks',
  ],
  icon: '🎨',
  keyTakeaways: [
    'Understand how diffusion models work and why they excel at image generation',
    'Master Stable Diffusion for local, customizable generation',
    'Use DALL-E 3 API for production-quality image generation',
    'Write effective prompts that produce consistent, high-quality results',
    'Transform images with img2img for style transfer and variations',
    'Apply ControlNet for precise structural control over generation',
    'Edit images with inpainting for seamless object removal and addition',
    'Generate and restore faces with specialized models like GFPGAN',
    'Upscale images with AI for genuine detail enhancement',
    'Analyze images with vision-capable LLMs (GPT-4V, Claude 3)',
    'Build complex workflows with ComfyUI for automation',
    'Deploy models on cloud platforms like Replicate',
    'Architect complete production image generation platforms',
  ],
  learningObjectives: [
    'Explain diffusion model architecture and the generation process',
    'Choose appropriate models (SD, SDXL, DALL-E) for different use cases',
    'Implement text-to-image generation with proper parameter control',
    'Engineer prompts with structure, weights, and negative prompts',
    'Apply img2img transformations with optimal strength values',
    'Use ControlNet for pose, depth, and edge-guided generation',
    'Perform inpainting for object removal, replacement, and outpainting',
    'Restore and enhance faces with GFPGAN and CodeFormer',
    'Upscale images using Real-ESRGAN and SD upscaling',
    'Extract information from images with vision LLMs',
    'Build automated workflows with ComfyUI and its API',
    'Deploy on cloud platforms with cost optimization',
    'Design production systems with queues, storage, and APIs',
  ],
  sections: [
    {
      ...imagegenerationfundamentalsSection,
      quiz: imagegenerationfundamentalsQuiz,
      multipleChoice: imagegenerationfundamentalsMultipleChoice,
    },
    {
      ...dalle3apiSection,
      quiz: dalle3apiQuiz,
      multipleChoice: dalle3apiMultipleChoice,
    },
    {
      ...stablediffusionSection,
      quiz: stablediffusionQuiz,
      multipleChoice: stablediffusionMultipleChoice,
    },
    {
      ...advancedpromptingimagesSection,
      quiz: advancedpromptingimagesQuiz,
      multipleChoice: advancedpromptingimagesMultipleChoice,
    },
    {
      ...imagetoimagegenerationSection,
      quiz: imagetoimagegenerationQuiz,
      multipleChoice: imagetoimagegenerationMultipleChoice,
    },
    {
      ...controlnetconditioningSection,
      quiz: controlnetconditioningQuiz,
      multipleChoice: controlnetconditioningMultipleChoice,
    },
    {
      ...inpaintingeditingSection,
      quiz: inpaintingeditingQuiz,
      multipleChoice: inpaintingeditingMultipleChoice,
    },
    {
      ...facegenerationrestorationSection,
      quiz: facegenerationrestorationQuiz,
      multipleChoice: facegenerationrestorationMultipleChoice,
    },
    {
      ...upscalingenhancementSection,
      quiz: upscalingenhancementQuiz,
      multipleChoice: upscalingenhancementMultipleChoice,
    },
    {
      ...computervisionllmsSection,
      quiz: computervisionllmsQuiz,
      multipleChoice: computervisionllmsMultipleChoice,
    },
    {
      ...comfyuiworkflowsSection,
      quiz: comfyuiworkflowsQuiz,
      multipleChoice: comfyuiworkflowsMultipleChoice,
    },
    {
      ...replicatemodelhostingSection,
      quiz: replicatemodelhostingQuiz,
      multipleChoice: replicatemodelhostingMultipleChoice,
    },
    {
      ...buildingimmagegenerationplatformSection,
      quiz: buildingimagegenerationplatformQuiz,
      multipleChoice: buildingimmagegenerationplatformMultipleChoice,
    },
  ],
};
