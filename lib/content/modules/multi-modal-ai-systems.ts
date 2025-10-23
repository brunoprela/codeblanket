/**
 * Multi-Modal AI Systems Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { multiModalFundamentals } from '../sections/multi-modal-ai-systems/multi-modal-fundamentals';
import { imageTextUnderstanding } from '../sections/multi-modal-ai-systems/image-text-understanding';
import { videoTextUnderstanding } from '../sections/multi-modal-ai-systems/video-text-understanding';
import { audioTextProcessing } from '../sections/multi-modal-ai-systems/audio-text-processing';
import { multiModalRag } from '../sections/multi-modal-ai-systems/multi-modal-rag';
import { crossModalGeneration } from '../sections/multi-modal-ai-systems/cross-modal-generation';
import { documentIntelligence } from '../sections/multi-modal-ai-systems/document-intelligence';
import { presentationSlideGeneration } from '../sections/multi-modal-ai-systems/presentation-slide-generation';
import { multiModalAgents } from '../sections/multi-modal-ai-systems/multi-modal-agents';
import { accessibilityApplications } from '../sections/multi-modal-ai-systems/accessibility-applications';
import { buildingMultiModalProducts } from '../sections/multi-modal-ai-systems/building-multi-modal-products';

// Import quizzes
import { multiModalFundamentalsQuiz } from '../quizzes/multi-modal-ai-systems/multi-modal-fundamentals';
import { imagetextunderstandingQuiz } from '../quizzes/multi-modal-ai-systems/image-text-understanding';
import { videotextunderstandingQuiz } from '../quizzes/multi-modal-ai-systems/video-text-understanding';
import { audiotextprocessingQuiz } from '../quizzes/multi-modal-ai-systems/audio-text-processing';
import { multimodalragQuiz } from '../quizzes/multi-modal-ai-systems/multi-modal-rag';
import { crossmodalgenerationQuiz } from '../quizzes/multi-modal-ai-systems/cross-modal-generation';
import { documentintelligenceQuiz } from '../quizzes/multi-modal-ai-systems/document-intelligence';
import { presentationslidegenerationQuiz } from '../quizzes/multi-modal-ai-systems/presentation-slide-generation';
import { multimodalagentsQuiz } from '../quizzes/multi-modal-ai-systems/multi-modal-agents';
import { accessibilityapplicationsQuiz } from '../quizzes/multi-modal-ai-systems/accessibility-applications';
import { buildingmultimodalproductsQuiz } from '../quizzes/multi-modal-ai-systems/building-multi-modal-products';

// Import multiple choice
import { multimodalfundamentalsMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/multi-modal-fundamentals';
import { imagetextunderstandingMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/image-text-understanding';
import { videotextunderstandingMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/video-text-understanding';
import { audiotextprocessingMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/audio-text-processing';
import { multimodalragMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/multi-modal-rag';
import { crossmodalgenerationMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/cross-modal-generation';
import { documentintelligenceMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/document-intelligence';
import { presentationslidegenerationMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/presentation-slide-generation';
import { multimodalagentsMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/multi-modal-agents';
import { accessibilityapplicationsMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/accessibility-applications';
import { buildingmultimodalproductsMultipleChoice } from '../multiple-choice/multi-modal-ai-systems/building-multi-modal-products';

export const multiModalAiSystemsModule: Module = {
  id: 'multi-modal-ai-systems',
  title: 'Multi-Modal AI Systems',
  description:
    'Master building sophisticated multi-modal AI systems that combine text, images, audio, and video. From understanding individual modalities to building complete products.',
  category: 'Applied AI',
  difficulty: 'Advanced',
  estimatedTime: '18 hours',
  prerequisites: [
    'LLM Engineering Fundamentals',
    'Prompt Engineering basics',
    'Python proficiency',
  ],
  icon: '🌐',
  keyTakeaways: [
    'Understand multi-modal AI fundamentals and architectures',
    'Build visual question answering systems',
    'Process and analyze video content',
    'Transcribe and analyze audio with Whisper',
    'Create multi-modal RAG systems with CLIP',
    'Generate content across modalities (text→image, image→text, etc.)',
    'Extract structured data from complex documents',
    'Auto-generate presentations with images and layouts',
    'Build agents that perceive and act across modalities',
    'Create accessible AI applications',
    'Design and deploy complete multi-modal products',
  ],
  learningObjectives: [
    'Master multi-modal AI architecture patterns',
    'Implement image captioning and visual question answering',
    'Build video understanding and summarization systems',
    'Process audio with speech-to-text and text-to-speech',
    'Create cross-modal retrieval systems with unified embeddings',
    'Generate images, videos, and audio from text prompts',
    'Process invoices, receipts, and forms automatically',
    'Generate professional presentations from content',
    'Build autonomous multi-modal agents',
    'Implement accessibility features with AI',
    'Architect scalable multi-modal products for production',
  ],
  sections: [
    {
      ...multiModalFundamentals,
      quiz: multiModalFundamentalsQuiz,
      multipleChoice: multimodalfundamentalsMultipleChoice,
    },
    {
      ...imageTextUnderstanding,
      quiz: imagetextunderstandingQuiz,
      multipleChoice: imagetextunderstandingMultipleChoice,
    },
    {
      ...videoTextUnderstanding,
      quiz: videotextunderstandingQuiz,
      multipleChoice: videotextunderstandingMultipleChoice,
    },
    {
      ...audioTextProcessing,
      quiz: audiotextprocessingQuiz,
      multipleChoice: audiotextprocessingMultipleChoice,
    },
    {
      ...multiModalRag,
      quiz: multimodalragQuiz,
      multipleChoice: multimodalragMultipleChoice,
    },
    {
      ...crossModalGeneration,
      quiz: crossmodalgenerationQuiz,
      multipleChoice: crossmodalgenerationMultipleChoice,
    },
    {
      ...documentIntelligence,
      quiz: documentintelligenceQuiz,
      multipleChoice: documentintelligenceMultipleChoice,
    },
    {
      ...presentationSlideGeneration,
      quiz: presentationslidegenerationQuiz,
      multipleChoice: presentationslidegenerationMultipleChoice,
    },
    {
      ...multiModalAgents,
      quiz: multimodalagentsQuiz,
      multipleChoice: multimodalagentsMultipleChoice,
    },
    {
      ...accessibilityApplications,
      quiz: accessibilityapplicationsQuiz,
      multipleChoice: accessibilityapplicationsMultipleChoice,
    },
    {
      ...buildingMultiModalProducts,
      quiz: buildingmultimodalproductsQuiz,
      multipleChoice: buildingmultimodalproductsMultipleChoice,
    },
  ],
};
