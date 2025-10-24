/**
 * Module: Building Complete AI Products
 * Module 15 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { productArchitectureDesign } from '../sections/building-complete-ai-products/product-architecture-design';
import { buildingAICodeEditor } from '../sections/building-complete-ai-products/building-ai-code-editor';
import { idePluginDevelopment } from '../sections/building-complete-ai-products/ide-plugin-development';
import { realTimeCollaboration } from '../sections/building-complete-ai-products/real-time-collaboration';
import { buildingAiResearchAssistant } from '../sections/building-complete-ai-products/building-ai-research-assistant';
import { buildingDocumentProcessingSystem } from '../sections/building-complete-ai-products/building-document-processing-system';
import { buildingMediaGenerationPlatform } from '../sections/building-complete-ai-products/building-media-generation-platform';
import { buildingConversationalAi } from '../sections/building-complete-ai-products/building-conversational-ai';
import { buildingAiPoweredExcelEditor } from '../sections/building-complete-ai-products/building-ai-powered-excel-editor';
import { buildingCursorForExcelFinance } from '../sections/building-complete-ai-products/building-cursor-for-excel-finance';
import { buildingFinancialAiApplications } from '../sections/building-complete-ai-products/building-financial-ai-applications';
import { frontendDevelopment } from '../sections/building-complete-ai-products/frontend-development';
import { backendDevelopment } from '../sections/building-complete-ai-products/backend-development';
import { devopsDeployment } from '../sections/building-complete-ai-products/devops-deployment';
import { productAnalyticsMetrics } from '../sections/building-complete-ai-products/product-analytics-metrics';
import { goToMarketStrategy } from '../sections/building-complete-ai-products/go-to-market-strategy';
import { puttingItAllTogether } from '../sections/building-complete-ai-products/putting-it-all-together';

// Quiz imports
import { productArchitectureDesignQuiz } from '../quizzes/building-complete-ai-products/product-architecture-design';
import { buildingAICodeEditorQuiz } from '../quizzes/building-complete-ai-products/building-ai-code-editor';
import { idePluginDevelopmentQuiz } from '../quizzes/building-complete-ai-products/ide-plugin-development';
import { realTimeCollaborationQuiz } from '../quizzes/building-complete-ai-products/real-time-collaboration';
import { buildingAiResearchAssistantQuiz } from '../quizzes/building-complete-ai-products/building-ai-research-assistant';
import { buildingDocumentProcessingSystemQuiz } from '../quizzes/building-complete-ai-products/building-document-processing-system';
import { buildingMediaGenerationPlatformQuiz } from '../quizzes/building-complete-ai-products/building-media-generation-platform';
import { buildingConversationalAiQuiz } from '../quizzes/building-complete-ai-products/building-conversational-ai';
import { buildingAiPoweredExcelEditorQuiz } from '../quizzes/building-complete-ai-products/building-ai-powered-excel-editor';
import { buildingCursorForExcelFinanceQuiz } from '../quizzes/building-complete-ai-products/building-cursor-for-excel-finance';
import { buildingFinancialAiApplicationsQuiz } from '../quizzes/building-complete-ai-products/building-financial-ai-applications';
import { frontendDevelopmentQuiz } from '../quizzes/building-complete-ai-products/frontend-development';
import { backendDevelopmentQuiz } from '../quizzes/building-complete-ai-products/backend-development';
import { devopsDeploymentQuiz } from '../quizzes/building-complete-ai-products/devops-deployment';
import { productAnalyticsMetricsQuiz } from '../quizzes/building-complete-ai-products/product-analytics-metrics';
import { goToMarketStrategyQuiz } from '../quizzes/building-complete-ai-products/go-to-market-strategy';
import { puttingItAllTogetherQuiz } from '../quizzes/building-complete-ai-products/putting-it-all-together';

// Multiple choice imports
import { productArchitectureDesignMultipleChoice } from '../multiple-choice/building-complete-ai-products/product-architecture-design';
import { buildingAiCodeEditorMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-ai-code-editor';
import { idePluginDevelopmentMultipleChoice } from '../multiple-choice/building-complete-ai-products/ide-plugin-development';
import { realTimeCollaborationMultipleChoice } from '../multiple-choice/building-complete-ai-products/real-time-collaboration';
import { buildingAiResearchAssistantMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-ai-research-assistant';
import { buildingDocumentProcessingSystemMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-document-processing-system';
import { buildingMediaGenerationPlatformMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-media-generation-platform';
import { buildingConversationalAiMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-conversational-ai';
import { buildingAiPoweredExcelEditorMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-ai-powered-excel-editor';
import { buildingCursorForExcelFinanceMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-cursor-for-excel-finance';
import { buildingFinancialAiApplicationsMultipleChoice } from '../multiple-choice/building-complete-ai-products/building-financial-ai-applications';
import { frontendDevelopmentMultipleChoice } from '../multiple-choice/building-complete-ai-products/frontend-development';
import { backendDevelopmentMultipleChoice } from '../multiple-choice/building-complete-ai-products/backend-development';
import { devopsDeploymentMultipleChoice } from '../multiple-choice/building-complete-ai-products/devops-deployment';
import { productAnalyticsMetricsMultipleChoice } from '../multiple-choice/building-complete-ai-products/product-analytics-metrics';
import { goToMarketStrategyMultipleChoice } from '../multiple-choice/building-complete-ai-products/go-to-market-strategy';
import { puttingItAllTogetherMultipleChoice } from '../multiple-choice/building-complete-ai-products/putting-it-all-together';

export const buildingCompleteAIProductsModule: Module = {
  id: 'applied-ai-complete-products',
  title: 'Building Complete AI Products',
  description:
    'Master building end-to-end AI products from concept to production. Learn to architect, develop, deploy, and scale complete AI applications including code editors, document processors, media platforms, and financial AI tools. Build production-ready systems like Cursor, with IDE plugins, real-time collaboration, and comprehensive deployment strategies.',
  icon: '🏗️',
  keyTakeaways: [
    'Design scalable architectures for production AI applications',
    'Build complete AI code editors with context management and diff generation',
    'Develop IDE plugins for VSCode and JetBrains with seamless integration',
    'Implement real-time collaboration with CRDTs and operational transformation',
    'Create multi-agent research assistants with web search and document processing',
    'Build document processing systems handling PDF, Excel, Word files',
    'Develop media generation platforms with queue management and GPU optimization',
    'Design conversational AI with memory, context, and tool integration',
    'Build Excel AI assistants and financial analysis applications',
    'Implement production frontend with React/Next.js and streaming responses',
    'Design FastAPI backends with authentication, caching, and background jobs',
    'Deploy with Docker, Kubernetes, and comprehensive monitoring',
    'Track product analytics, user behavior, and LLM-specific metrics',
    'Execute go-to-market strategies with pricing, marketing, and user onboarding',
    'Integrate all components into cohesive, production-ready AI products',
    'Optimize performance, cost, and user experience at scale',
    'Handle security, compliance, and data privacy in production',
  ],
  sections: [
    {
      ...productArchitectureDesign,
      quiz: productArchitectureDesignQuiz,
      multipleChoice: productArchitectureDesignMultipleChoice,
    },
    {
      ...buildingAICodeEditor,
      quiz: buildingAICodeEditorQuiz,
      multipleChoice: buildingAiCodeEditorMultipleChoice,
    },
    {
      ...idePluginDevelopment,
      quiz: idePluginDevelopmentQuiz,
      multipleChoice: idePluginDevelopmentMultipleChoice,
    },
    {
      ...realTimeCollaboration,
      quiz: realTimeCollaborationQuiz,
      multipleChoice: realTimeCollaborationMultipleChoice,
    },
    {
      ...buildingAiResearchAssistant,
      quiz: buildingAiResearchAssistantQuiz,
      multipleChoice: buildingAiResearchAssistantMultipleChoice,
    },
    {
      ...buildingDocumentProcessingSystem,
      quiz: buildingDocumentProcessingSystemQuiz,
      multipleChoice: buildingDocumentProcessingSystemMultipleChoice,
    },
    {
      ...buildingMediaGenerationPlatform,
      quiz: buildingMediaGenerationPlatformQuiz,
      multipleChoice: buildingMediaGenerationPlatformMultipleChoice,
    },
    {
      ...buildingConversationalAi,
      quiz: buildingConversationalAiQuiz,
      multipleChoice: buildingConversationalAiMultipleChoice,
    },
    {
      ...buildingAiPoweredExcelEditor,
      quiz: buildingAiPoweredExcelEditorQuiz,
      multipleChoice: buildingAiPoweredExcelEditorMultipleChoice,
    },
    {
      ...buildingCursorForExcelFinance,
      quiz: buildingCursorForExcelFinanceQuiz,
      multipleChoice: buildingCursorForExcelFinanceMultipleChoice,
    },
    {
      ...buildingFinancialAiApplications,
      quiz: buildingFinancialAiApplicationsQuiz,
      multipleChoice: buildingFinancialAiApplicationsMultipleChoice,
    },
    {
      ...frontendDevelopment,
      quiz: frontendDevelopmentQuiz,
      multipleChoice: frontendDevelopmentMultipleChoice,
    },
    {
      ...backendDevelopment,
      quiz: backendDevelopmentQuiz,
      multipleChoice: backendDevelopmentMultipleChoice,
    },
    {
      ...devopsDeployment,
      quiz: devopsDeploymentQuiz,
      multipleChoice: devopsDeploymentMultipleChoice,
    },
    {
      ...productAnalyticsMetrics,
      quiz: productAnalyticsMetricsQuiz,
      multipleChoice: productAnalyticsMetricsMultipleChoice,
    },
    {
      ...goToMarketStrategy,
      quiz: goToMarketStrategyQuiz,
      multipleChoice: goToMarketStrategyMultipleChoice,
    },
    {
      ...puttingItAllTogether,
      quiz: puttingItAllTogetherQuiz,
      multipleChoice: puttingItAllTogetherMultipleChoice,
    },
  ],
};
