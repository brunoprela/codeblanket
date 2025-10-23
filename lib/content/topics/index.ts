/**
 * Topics Index
 * Exports all topic definitions and provides backward-compatible API
 */

import { Module } from '../../types';

// Import all topics
export * from './python';
export * from './system-design';
export * from './algorithms-data-structures';
export * from './quantitative-programming';
export * from './devops';
export * from './frontend-development';
export * from './behavioral';
export * from './applied-ai';
export * from './product-management';
export * from './engineering-management';
export * from './finance';

// Import topics for aggregation
import {
  pythonTopic,
  algorithmsDataStructuresTopic,
  systemDesignTopic,
  quantitativeProgrammingTopic,
  devopsTopic,
  frontendDevelopmentTopic,
  behavioralTopic,
  appliedAiTopic,
  productManagementTopic,
  engineeringManagementTopic,
  financeTopic,
} from '.';

// Import all modules
import { pythonFundamentalsModule } from '../modules/python-fundamentals';
import { pythonIntermediateModule } from '../modules/python-intermediate';
import { pythonAdvancedModule } from '../modules/python-advanced';
import { pythonOopModule } from '../modules/python-oop';

import { timeSpaceComplexityModule } from '../modules/time-space-complexity';
import { arraysHashingModule } from '../modules/arrays-hashing';
import { twoPointersModule } from '../modules/two-pointers';
import { slidingWindowModule } from '../modules/sliding-window';
import { dfsModule } from '../modules/dfs';
import { bfsModule } from '../modules/bfs';
import { binarySearchModule } from '../modules/binary-search';
import { sortingModule } from '../modules/sorting';
import { recursionModule } from '../modules/recursion';
import { stackModule } from '../modules/stack';
import { queueModule } from '../modules/queue';
import { designProblemsModule } from '../modules/design-problems';
import { linkedListModule } from '../modules/linked-list';
import { stringAlgorithmsModule } from '../modules/string-algorithms';
import { treesModule } from '../modules/trees';
import { heapModule } from '../modules/heap';
import { graphsModule } from '../modules/graphs';
import { backtrackingModule } from '../modules/backtracking';
import { dynamicProgrammingModule } from '../modules/dynamic-programming';
import { triesModule } from '../modules/tries';
import { intervalsModule } from '../modules/intervals';
import { greedyModule } from '../modules/greedy';
import { advancedGraphsModule } from '../modules/advanced-graphs';
import { segmentTreeModule } from '../modules/segment-tree';
import { fenwickTreeModule } from '../modules/fenwick-tree';
import { bitManipulationModule } from '../modules/bit-manipulation';
import { mathGeometryModule } from '../modules/math-geometry';

import { systemDesignFundamentalsModule } from '../modules/system-design-fundamentals';
import { systemDesignCoreBuildingBlocksModule } from '../modules/system-design-core-building-blocks';
import { systemDesignDatabaseDesignModule } from '../modules/system-design-database-design';
import { systemDesignNetworkingModule } from '../modules/system-design-networking';
import { systemDesignApiDesignModule } from '../modules/system-design-api-design';
import { systemDesignTradeoffsModule } from '../modules/system-design-tradeoffs';
import { systemDesignAuthenticationModule } from '../modules/system-design-authentication';
import { systemDesignMicroservicesModule } from '../modules/system-design-microservices';

import { mlMathematicalFoundationsModule } from '../modules/ml-mathematical-foundations';
import { mlCalculusFundamentalsModule } from '../modules/ml-calculus-fundamentals';
import { mlLinearAlgebraFoundationsModule } from '../modules/ml-linear-algebra-foundations';
import { mlProbabilityTheoryModule } from '../modules/ml-probability-theory';
import { mlStatisticsFundamentalsModule } from '../modules/ml-statistics-fundamentals';
import { mlPythonForDataScience } from '../modules/ml-python-for-data-science';
import { mlEdaFeatureEngineeringModule } from '../modules/ml-eda-feature-engineering';
import { mlSupervisedLearningModule } from '../modules/ml-supervised-learning';
import { mlUnsupervisedLearning } from '../modules/ml-unsupervised-learning';
import { mlDeepLearningFundamentals } from '../modules/ml-deep-learning-fundamentals';
import { mlAdvancedDeepLearningModule } from '../modules/ml-advanced-deep-learning';
import { naturalLanguageProcessingModule } from '../modules/ml-natural-language-processing';

import { promptEngineeringOptimizationModule } from '../modules/prompt-engineering-optimization';
import { llmEngineeringFundamentalsModule } from '../modules/llm-engineering-fundamentals';
import { fileProcessingDocumentUnderstandingModule } from '../modules/file-processing-document-understanding';
import { codeUnderstandingAstManipulationModule } from '../modules/code-understanding-ast-manipulation';
import { buildingCodeGenerationSystemsModule } from '../modules/building-code-generation-systems';
import { llmToolUseFunctionCallingModule } from '../modules/llm-tool-use-function-calling';
import { multiAgentSystemsOrchestrationModule } from '../modules/multi-agent-systems-orchestration';
import { imageGenerationComputerVisionModule } from '../modules/image-generation-computer-vision';
import { videoAudioGenerationModule } from '../modules/video-audio-generation';
import { multiModalAiSystemsModule } from '../modules/multi-modal-ai-systems';

// Create a map of all modules by ID
const allModulesMap: Record<string, Module> = {
  'python-fundamentals': pythonFundamentalsModule,
  'python-intermediate': pythonIntermediateModule,
  'python-advanced': pythonAdvancedModule,
  'python-oop': pythonOopModule,

  'time-space-complexity': timeSpaceComplexityModule,
  'arrays-hashing': arraysHashingModule,
  'two-pointers': twoPointersModule,
  'sliding-window': slidingWindowModule,
  dfs: dfsModule,
  bfs: bfsModule,
  'binary-search': binarySearchModule,
  sorting: sortingModule,
  recursion: recursionModule,
  stack: stackModule,
  queue: queueModule,
  'design-problems': designProblemsModule,
  'linked-list': linkedListModule,
  'string-algorithms': stringAlgorithmsModule,
  trees: treesModule,
  heap: heapModule,
  graphs: graphsModule,
  backtracking: backtrackingModule,
  'dynamic-programming': dynamicProgrammingModule,
  tries: triesModule,
  intervals: intervalsModule,
  greedy: greedyModule,
  'advanced-graphs': advancedGraphsModule,
  'segment-tree': segmentTreeModule,
  'fenwick-tree': fenwickTreeModule,
  'bit-manipulation': bitManipulationModule,
  'math-geometry': mathGeometryModule,

  'system-design-fundamentals': systemDesignFundamentalsModule,
  'system-design-core-building-blocks': systemDesignCoreBuildingBlocksModule,
  'system-design-database-design': systemDesignDatabaseDesignModule,
  'system-design-networking': systemDesignNetworkingModule,
  'system-design-api-design': systemDesignApiDesignModule,
  'system-design-tradeoffs': systemDesignTradeoffsModule,
  'system-design-authentication': systemDesignAuthenticationModule,
  'system-design-microservices': systemDesignMicroservicesModule,

  'ml-mathematical-foundations': mlMathematicalFoundationsModule,
  'ml-calculus-fundamentals': mlCalculusFundamentalsModule,
  'ml-linear-algebra-foundations': mlLinearAlgebraFoundationsModule,
  'ml-probability-theory': mlProbabilityTheoryModule,
  'ml-statistics-fundamentals': mlStatisticsFundamentalsModule,
  'ml-python-for-data-science': mlPythonForDataScience,
  'ml-eda-feature-engineering': mlEdaFeatureEngineeringModule,
  'ml-supervised-learning': mlSupervisedLearningModule,
  'ml-unsupervised-learning': mlUnsupervisedLearning,
  'ml-deep-learning-fundamentals': mlDeepLearningFundamentals,
  'ml-advanced-deep-learning': mlAdvancedDeepLearningModule,
  'ml-natural-language-processing': naturalLanguageProcessingModule,

  'applied-ai-llm-fundamentals': llmEngineeringFundamentalsModule,
  'applied-ai-prompt-engineering': promptEngineeringOptimizationModule,
  'applied-ai-file-processing': fileProcessingDocumentUnderstandingModule,
  'applied-ai-code-understanding': codeUnderstandingAstManipulationModule,
  'applied-ai-code-generation': buildingCodeGenerationSystemsModule,
  'applied-ai-tool-use': llmToolUseFunctionCallingModule,
  'applied-ai-multi-agent': multiAgentSystemsOrchestrationModule,
  'applied-ai-image-generation': imageGenerationComputerVisionModule,
  'applied-ai-video-audio': videoAudioGenerationModule,
  'applied-ai-multi-modal': multiModalAiSystemsModule,
};

// Define all topics in order
const allTopics = [
  frontendDevelopmentTopic,
  pythonTopic,
  algorithmsDataStructuresTopic,
  quantitativeProgrammingTopic,
  appliedAiTopic,
  devopsTopic,
  systemDesignTopic,
  productManagementTopic,
  engineeringManagementTopic,
  behavioralTopic,
  financeTopic,
];

// Build moduleCategories for backward compatibility
// Each module becomes its own category with a single 'module' property
export const moduleCategories = allTopics.flatMap((topic) =>
  topic.modules
    .map((moduleId) => {
      const moduleData = allModulesMap[moduleId];
      if (!moduleData) return null;
      return {
        id: moduleData.id,
        title: moduleData.title,
        description: moduleData.description,
        icon: moduleData.icon,
        module: moduleData,
      };
    })
    .filter((m): m is NonNullable<typeof m> => m !== null),
);

// Build topicSections for backward compatibility
// This groups modules by topic, where each topic has an array of module categories
export const topicSections = allTopics.map((topic) => ({
  id: topic.id,
  title: topic.title,
  icon: topic.icon,
  modules: topic.modules
    .map((moduleId) => {
      const moduleData = allModulesMap[moduleId];
      if (!moduleData) return null;
      return {
        id: moduleData.id,
        title: moduleData.title,
        description: moduleData.description,
        icon: moduleData.icon,
        module: moduleData,
      };
    })
    .filter((m): m is NonNullable<typeof m> => m !== null),
}));

// Export helper to get a module by ID
export const getModuleById = (id: string): Module | undefined => {
  return allModulesMap[id];
};

// Export all modules as an array
export const allModules = Object.values(allModulesMap);
