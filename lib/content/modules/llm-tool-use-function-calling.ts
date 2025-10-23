/**
 * LLM Tool Use & Function Calling Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { functionCallingFundamentals } from '../sections/llm-tool-use-function-calling/function-calling-fundamentals';
import { definingFunctionsTools } from '../sections/llm-tool-use-function-calling/defining-functions-tools';
import { functionCallingWorkflows } from '../sections/llm-tool-use-function-calling/function-calling-workflows';
import { buildingToolLibraries } from '../sections/llm-tool-use-function-calling/building-tool-libraries';
import { apiIntegrationTools } from '../sections/llm-tool-use-function-calling/api-integration-tools';
import { codeExecutionTools } from '../sections/llm-tool-use-function-calling/code-execution-tools';
import { toolUsePrompting } from '../sections/llm-tool-use-function-calling/tool-use-prompting';
import { structuredToolResponses } from '../sections/llm-tool-use-function-calling/structured-tool-responses';
import { toolUseObservability } from '../sections/llm-tool-use-function-calling/tool-use-observability';
import { advancedToolPatterns } from '../sections/llm-tool-use-function-calling/advanced-tool-patterns';
import { buildingAgenticSystem } from '../sections/llm-tool-use-function-calling/building-agentic-system';

// Import quizzes
import { functionCallingFundamentalsQuiz } from '../quizzes/llm-tool-use-function-calling/function-calling-fundamentals';
import { definingFunctionsToolsQuiz } from '../quizzes/llm-tool-use-function-calling/defining-functions-tools';
import { functionCallingWorkflowsQuiz } from '../quizzes/llm-tool-use-function-calling/function-calling-workflows';
import { buildingToolLibrariesQuiz } from '../quizzes/llm-tool-use-function-calling/building-tool-libraries';
import { apiIntegrationToolsQuiz } from '../quizzes/llm-tool-use-function-calling/api-integration-tools';
import { codeExecutionToolsQuiz } from '../quizzes/llm-tool-use-function-calling/code-execution-tools';
import { toolUsePromptingQuiz } from '../quizzes/llm-tool-use-function-calling/tool-use-prompting';
import { structuredToolResponsesQuiz } from '../quizzes/llm-tool-use-function-calling/structured-tool-responses';
import { toolUseObservabilityQuiz } from '../quizzes/llm-tool-use-function-calling/tool-use-observability';
import { advancedToolPatternsQuiz } from '../quizzes/llm-tool-use-function-calling/advanced-tool-patterns';
import { buildingAgenticSystemQuiz } from '../quizzes/llm-tool-use-function-calling/building-agentic-system';

// Import multiple choice
import { functionCallingFundamentalsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/function-calling-fundamentals';
import { definingFunctionsToolsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/defining-functions-tools';
import { functionCallingWorkflowsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/function-calling-workflows';
import { buildingToolLibrariesMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/building-tool-libraries';
import { apiIntegrationToolsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/api-integration-tools';
import { codeExecutionToolsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/code-execution-tools';
import { toolUsePromptingMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/tool-use-prompting';
import { structuredToolResponsesMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/structured-tool-responses';
import { toolUseObservabilityMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/tool-use-observability';
import { advancedToolPatternsMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/advanced-tool-patterns';
import { buildingAgenticSystemMultipleChoice } from '../multiple-choice/llm-tool-use-function-calling/building-agentic-system';

export const llmToolUseFunctionCallingModule: Module = {
  id: 'applied-ai-tool-use',
  title: 'LLM Tool Use & Function Calling',
  description:
    'Master function calling, tool use, and building LLM agents that interact with external systems. Learn how ChatGPT, Claude, and other AI assistants use tools to perform actions beyond text generation.',
  category: 'Applied AI',
  difficulty: 'Intermediate',
  estimatedTime: '2-3 weeks',
  prerequisites: [
    'Module 1: LLM Engineering Fundamentals',
    'Module 2: Prompt Engineering & Optimization',
    'Basic understanding of JSON Schema',
    'Python programming with async/await',
  ],
  icon: '🔧',
  keyTakeaways: [
    'Function calling allows LLMs to interact with external systems and perform actions beyond text generation',
    'Proper tool schemas with clear descriptions are crucial for reliable function calling',
    'Different workflow patterns (sequential, parallel, conditional) suit different use cases',
    'Tool libraries should be well-organized, documented, and easy to extend',
    'Security and sandboxing are critical when building code execution tools',
    'Effective prompting teaches the LLM when and how to use tools appropriately',
    'Structured responses make it easier for LLMs to process tool outputs',
    'Observability is essential for debugging and optimizing tool-using systems',
    'Advanced patterns like tool chaining enable complex multi-step workflows',
    'Production agentic systems require careful error handling, retry logic, and human oversight',
  ],
  learningObjectives: [
    'Understand how function calling works in OpenAI, Claude, and other LLM providers',
    'Design and implement tool schemas that LLMs can reliably use',
    'Build various function calling workflows for different use cases',
    'Create reusable tool libraries with proper organization and documentation',
    'Integrate external APIs as tools for LLM consumption',
    'Implement safe code execution environments like ChatGPT Code Interpreter',
    'Master prompt engineering techniques for effective tool use',
    'Design tool responses that maximize LLM understanding',
    'Implement comprehensive monitoring and debugging for tool systems',
    'Apply advanced patterns for complex tool orchestration',
    'Build production-ready agentic systems with proper guardrails',
  ],
  sections: [
    {
      ...functionCallingFundamentals,
      quiz: functionCallingFundamentalsQuiz,
      multipleChoice: functionCallingFundamentalsMultipleChoice,
    },
    {
      ...definingFunctionsTools,
      quiz: definingFunctionsToolsQuiz,
      multipleChoice: definingFunctionsToolsMultipleChoice,
    },
    {
      ...functionCallingWorkflows,
      quiz: functionCallingWorkflowsQuiz,
      multipleChoice: functionCallingWorkflowsMultipleChoice,
    },
    {
      ...buildingToolLibraries,
      quiz: buildingToolLibrariesQuiz,
      multipleChoice: buildingToolLibrariesMultipleChoice,
    },
    {
      ...apiIntegrationTools,
      quiz: apiIntegrationToolsQuiz,
      multipleChoice: apiIntegrationToolsMultipleChoice,
    },
    {
      ...codeExecutionTools,
      quiz: codeExecutionToolsQuiz,
      multipleChoice: codeExecutionToolsMultipleChoice,
    },
    {
      ...toolUsePrompting,
      quiz: toolUsePromptingQuiz,
      multipleChoice: toolUsePromptingMultipleChoice,
    },
    {
      ...structuredToolResponses,
      quiz: structuredToolResponsesQuiz,
      multipleChoice: structuredToolResponsesMultipleChoice,
    },
    {
      ...toolUseObservability,
      quiz: toolUseObservabilityQuiz,
      multipleChoice: toolUseObservabilityMultipleChoice,
    },
    {
      ...advancedToolPatterns,
      quiz: advancedToolPatternsQuiz,
      multipleChoice: advancedToolPatternsMultipleChoice,
    },
    {
      ...buildingAgenticSystem,
      quiz: buildingAgenticSystemQuiz,
      multipleChoice: buildingAgenticSystemMultipleChoice,
    },
  ],
};
