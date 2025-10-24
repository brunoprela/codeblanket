/**
 * Module: Large Language Models (LLMs)
 * Module 14 of ML & AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { llmFundamentals } from '../sections/large-language-models/llm-fundamentals';
import { transformerArchitectureDeepDive } from '../sections/large-language-models/transformer-architecture-deep-dive';
import { gptFamily } from '../sections/large-language-models/gpt-family';
import { bertEncoderModels } from '../sections/large-language-models/bert-encoder-models';
import { llmTrainingProcess } from '../sections/large-language-models/llm-training-process';
import { finetuningStrategies } from '../sections/large-language-models/finetuning-strategies';
import { promptEngineering } from '../sections/large-language-models/prompt-engineering';
import { llmAlignmentRLHF } from '../sections/large-language-models/llm-alignment-rlhf';
import { vectorDatabasesEmbeddings } from '../sections/large-language-models/vector-databases-embeddings';
import { retrievalAugmentedGeneration } from '../sections/large-language-models/retrieval-augmented-generation';
import { llmAgentsToolUse } from '../sections/large-language-models/llm-agents-tool-use';
import { contextWindowManagement } from '../sections/large-language-models/context-window-management';
import { advancedLLMArchitectures } from '../sections/large-language-models/advanced-llm-architectures';
import { llmEvaluationSafety } from '../sections/large-language-models/llm-evaluation-safety';
import { llmCostOptimization } from '../sections/large-language-models/llm-cost-optimization';
import { llmDeploymentProduction } from '../sections/large-language-models/llm-deployment-production';

// Quiz imports
import { llmFundamentalsQuiz } from '../quizzes/large-language-models/llm-fundamentals';
import { transformerArchitectureQuiz } from '../quizzes/large-language-models/transformer-architecture-deep-dive';
import { gptFamilyQuiz } from '../quizzes/large-language-models/gpt-family';
import { bertEncoderModelsQuiz } from '../quizzes/large-language-models/bert-encoder-models';
import { llmTrainingProcessQuiz } from '../quizzes/large-language-models/llm-training-process';
import { finetuningStrategiesQuiz } from '../quizzes/large-language-models/finetuning-strategies';
import { promptEngineeringQuiz } from '../quizzes/large-language-models/prompt-engineering';
import { llmAlignmentRLHFQuiz } from '../quizzes/large-language-models/llm-alignment-rlhf';
import { vectorDatabasesEmbeddingsQuiz } from '../quizzes/large-language-models/vector-databases-embeddings';
import { retrievalAugmentedGenerationQuiz } from '../quizzes/large-language-models/retrieval-augmented-generation';
import { llmAgentsToolUseQuiz } from '../quizzes/large-language-models/llm-agents-tool-use';
import { contextWindowManagementQuiz } from '../quizzes/large-language-models/context-window-management';
import { advancedLLMArchitecturesQuiz } from '../quizzes/large-language-models/advanced-llm-architectures';
import { llmEvaluationSafetyQuiz } from '../quizzes/large-language-models/llm-evaluation-safety';
import { llmCostOptimizationQuiz } from '../quizzes/large-language-models/llm-cost-optimization';
import { llmDeploymentProductionQuiz } from '../quizzes/large-language-models/llm-deployment-production';

// Multiple choice imports
import { llmFundamentalsMC } from '../multiple-choice/large-language-models/llm-fundamentals';
import { transformerArchitectureMC } from '../multiple-choice/large-language-models/transformer-architecture-deep-dive';
import { gptFamilyMC } from '../multiple-choice/large-language-models/gpt-family';
import { bertEncoderModelsMC } from '../multiple-choice/large-language-models/bert-encoder-models';
import { llmTrainingProcessMC } from '../multiple-choice/large-language-models/llm-training-process';
import { finetuningStrategiesMC } from '../multiple-choice/large-language-models/finetuning-strategies';
import { promptEngineeringMC } from '../multiple-choice/large-language-models/prompt-engineering';
import { llmAlignmentRLHFMC } from '../multiple-choice/large-language-models/llm-alignment-rlhf';
import { vectorDatabasesEmbeddingsMC } from '../multiple-choice/large-language-models/vector-databases-embeddings';
import { retrievalAugmentedGenerationMC } from '../multiple-choice/large-language-models/retrieval-augmented-generation';
import { llmAgentsToolUseMC } from '../multiple-choice/large-language-models/llm-agents-tool-use';
import { contextWindowManagementMC } from '../multiple-choice/large-language-models/context-window-management';
import { advancedLLMArchitecturesMC } from '../multiple-choice/large-language-models/advanced-llm-architectures';
import { llmEvaluationSafetyMC } from '../multiple-choice/large-language-models/llm-evaluation-safety';
import { llmCostOptimizationMC } from '../multiple-choice/large-language-models/llm-cost-optimization';
import { llmDeploymentProductionMC } from '../multiple-choice/large-language-models/llm-deployment-production';

// Helper to transform quiz format from large-language-models format to standard format
const transformQuiz = (
  quiz: { id: number; question: string; expectedAnswer: string }[],
) => {
  return quiz.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    sampleAnswer: q.expectedAnswer,
    keyPoints: [], // LLM quizzes don't have keyPoints, so we'll use empty array
  }));
};

// Helper to transform multiple choice format
const transformMC = (
  mc: {
    id: number;
    question: string;
    options: string[];
    correctAnswer: number;
    explanation: string;
  }[],
) => {
  return mc.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    options: q.options,
    correctAnswer: q.correctAnswer,
    explanation: q.explanation,
  }));
};

export const largeLanguageModelsModule: Module = {
  id: 'ml-ai-large-language-models',
  title: 'Large Language Models (LLMs)',
  description:
    'Master modern LLM architectures, training, fine-tuning, agents, tools, and production deployment. Learn transformer architectures in depth, train and fine-tune models efficiently with LoRA and PEFT, engineer effective prompts, build RAG systems, create autonomous agents with tool use, manage context windows, optimize costs, and deploy scalable LLM applications. Build production-ready systems like ChatGPT with proper evaluation, safety guardrails, and monitoring.',
  icon: '🤖',
  keyTakeaways: [
    'Understand transformer architecture and attention mechanisms in depth',
    'Master GPT and BERT model families and their applications',
    'Train LLMs from scratch with distributed training and mixed precision',
    'Fine-tune models efficiently using LoRA, QLoRA, and parameter-efficient methods',
    'Engineer effective prompts with zero-shot, few-shot, and chain-of-thought techniques',
    'Implement RLHF and alignment techniques for safe, helpful models',
    'Build vector databases and semantic search with embeddings',
    'Create production RAG systems with chunking, retrieval, and reranking',
    'Develop LLM agents with tool use, memory, and ReAct patterns',
    'Manage context windows for long documents with compression and chunking',
    'Work with advanced architectures like MoE and multimodal models',
    'Evaluate LLMs with benchmarks, human evaluation, and safety metrics',
    'Optimize costs with caching, quantization, and efficient inference',
    'Deploy scalable LLM systems with vLLM, streaming, and monitoring',
    'Handle hallucinations, bias, and safety in production',
    'Build complete LLM applications from research to deployment',
  ],
  sections: [
    {
      ...llmFundamentals,
      quiz: transformQuiz(llmFundamentalsQuiz.questions),
      multipleChoice: transformMC(llmFundamentalsMC.questions),
    },
    {
      ...transformerArchitectureDeepDive,
      quiz: transformQuiz(transformerArchitectureQuiz.questions),
      multipleChoice: transformMC(transformerArchitectureMC.questions),
    },
    {
      ...gptFamily,
      quiz: transformQuiz(gptFamilyQuiz.questions),
      multipleChoice: transformMC(gptFamilyMC.questions),
    },
    {
      ...bertEncoderModels,
      quiz: transformQuiz(bertEncoderModelsQuiz.questions),
      multipleChoice: transformMC(bertEncoderModelsMC.questions),
    },
    {
      ...llmTrainingProcess,
      quiz: transformQuiz(llmTrainingProcessQuiz.questions),
      multipleChoice: transformMC(llmTrainingProcessMC.questions),
    },
    {
      ...finetuningStrategies,
      quiz: transformQuiz(finetuningStrategiesQuiz.questions),
      multipleChoice: transformMC(finetuningStrategiesMC.questions),
    },
    {
      ...promptEngineering,
      quiz: transformQuiz(promptEngineeringQuiz.questions),
      multipleChoice: transformMC(promptEngineeringMC.questions),
    },
    {
      ...llmAlignmentRLHF,
      quiz: transformQuiz(llmAlignmentRLHFQuiz.questions),
      multipleChoice: transformMC(llmAlignmentRLHFMC.questions),
    },
    {
      ...vectorDatabasesEmbeddings,
      quiz: transformQuiz(vectorDatabasesEmbeddingsQuiz.questions),
      multipleChoice: transformMC(vectorDatabasesEmbeddingsMC.questions),
    },
    {
      ...retrievalAugmentedGeneration,
      quiz: transformQuiz(retrievalAugmentedGenerationQuiz.questions),
      multipleChoice: transformMC(retrievalAugmentedGenerationMC.questions),
    },
    {
      ...llmAgentsToolUse,
      quiz: transformQuiz(llmAgentsToolUseQuiz.questions),
      multipleChoice: transformMC(llmAgentsToolUseMC.questions),
    },
    {
      ...contextWindowManagement,
      quiz: transformQuiz(contextWindowManagementQuiz.questions),
      multipleChoice: transformMC(contextWindowManagementMC.questions),
    },
    {
      ...advancedLLMArchitectures,
      quiz: transformQuiz(advancedLLMArchitecturesQuiz.questions),
      multipleChoice: transformMC(advancedLLMArchitecturesMC.questions),
    },
    {
      ...llmEvaluationSafety,
      quiz: transformQuiz(llmEvaluationSafetyQuiz.questions),
      multipleChoice: transformMC(llmEvaluationSafetyMC.questions),
    },
    {
      ...llmCostOptimization,
      quiz: transformQuiz(llmCostOptimizationQuiz.questions),
      multipleChoice: transformMC(llmCostOptimizationMC.questions),
    },
    {
      ...llmDeploymentProduction,
      quiz: transformQuiz(llmDeploymentProductionQuiz.questions),
      multipleChoice: transformMC(llmDeploymentProductionMC.questions),
    },
  ],
};
