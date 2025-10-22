/**
 * LLM Engineering Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { llmapisprovidersSection } from '../sections/llm-engineering-fundamentals/llm-apis-providers';
import { chatcompletionsmessagesSection } from '../sections/llm-engineering-fundamentals/chat-completions-messages';
import { tokenscontextwindowsSection } from '../sections/llm-engineering-fundamentals/tokens-context-windows';
import { temperaturesamplingSection } from '../sections/llm-engineering-fundamentals/temperature-sampling';
import { streamingresponsesSection } from '../sections/llm-engineering-fundamentals/streaming-responses';
import { errorhandlingretrySection } from '../sections/llm-engineering-fundamentals/error-handling-retry';
import { costtrackingoptimizationSection } from '../sections/llm-engineering-fundamentals/cost-tracking-optimization';
import { prompttemplatesSection } from '../sections/llm-engineering-fundamentals/prompt-templates';
import { outputparsingSection } from '../sections/llm-engineering-fundamentals/output-parsing';
import { observabilityloggingSection } from '../sections/llm-engineering-fundamentals/observability-logging';
import { cachingperformanceSection } from '../sections/llm-engineering-fundamentals/caching-performance';
import { localllmdeploymentSection } from '../sections/llm-engineering-fundamentals/local-llm-deployment';

// Import quizzes
import { llmapisprovidersQuiz } from '../quizzes/llm-engineering-fundamentals/llm-apis-providers';
import { chatcompletionsmessagesQuiz } from '../quizzes/llm-engineering-fundamentals/chat-completions-messages';
import { tokenscontextwindowsQuiz } from '../quizzes/llm-engineering-fundamentals/tokens-context-windows';
import { temperaturesamplingQuiz } from '../quizzes/llm-engineering-fundamentals/temperature-sampling';
import { streamingresponsesQuiz } from '../quizzes/llm-engineering-fundamentals/streaming-responses';
import { errorhandlingretryQuiz } from '../quizzes/llm-engineering-fundamentals/error-handling-retry';
import { costtrackingoptimizationQuiz } from '../quizzes/llm-engineering-fundamentals/cost-tracking-optimization';
import { prompttemplatesQuiz } from '../quizzes/llm-engineering-fundamentals/prompt-templates';
import { outputparsingQuiz } from '../quizzes/llm-engineering-fundamentals/output-parsing';
import { observabilityloggingQuiz } from '../quizzes/llm-engineering-fundamentals/observability-logging';
import { cachingperformanceQuiz } from '../quizzes/llm-engineering-fundamentals/caching-performance';
import { localllmdeploymentQuiz } from '../quizzes/llm-engineering-fundamentals/local-llm-deployment';

// Import multiple choice
import { llmapisprovidersMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/llm-apis-providers';
import { chatcompletionsmessagesMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/chat-completions-messages';
import { tokenscontextwindowsMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/tokens-context-windows';
import { temperaturesamplingMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/temperature-sampling';
import { streamingresponsesMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/streaming-responses';
import { errorhandlingretryMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/error-handling-retry';
import { costtrackingoptimizationMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/cost-tracking-optimization';
import { prompttemplatesMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/prompt-templates';
import { outputparsingMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/output-parsing';
import { observabilityloggingMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/observability-logging';
import { cachingperformanceMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/caching-performance';
import { localllmdeploymentMultipleChoice } from '../multiple-choice/llm-engineering-fundamentals/local-llm-deployment';

export const llmEngineeringFundamentalsModule: Module = {
    id: 'applied-ai-llm-fundamentals',
    title: 'LLM Engineering Fundamentals',
    description:
        'Master the essential skills for building production-ready LLM applications: from API basics to advanced optimization, cost management, and observability.',
    category: 'Applied AI',
    difficulty: 'Intermediate',
    estimatedTime: '12 hours',
    prerequisites: ['Python fundamentals', 'Basic API understanding'],
    icon: '🤖',
    keyTakeaways: [
        'Make reliable LLM API calls with proper error handling',
        'Understand tokens, context windows, and cost implications',
        'Implement streaming for better user experience',
        'Build robust retry logic with exponential backoff',
        'Track and optimize LLM costs effectively',
        'Parse structured data from LLM outputs',
        'Implement comprehensive logging and observability',
        'Deploy local LLMs when economically viable',
    ],
    learningObjectives: [
        'Master LLM API calls with OpenAI, Anthropic, and other providers',
        'Manage conversation context and message formatting',
        'Understand token counting and context window limitations',
        'Control output with temperature, top_p, and sampling parameters',
        'Implement streaming responses for better UX',
        'Build production-grade error handling and retry logic',
        'Track, analyze, and optimize LLM costs',
        'Use prompt templates for maintainable applications',
        'Extract structured data using JSON mode and Pydantic',
        'Implement comprehensive observability and logging',
        'Optimize performance with caching strategies',
        'Evaluate and deploy local LLMs when appropriate',
    ],
    sections: [
        {
            ...llmapisprovidersSection,
            quiz: llmapisprovidersQuiz,
            multipleChoice: llmapisprovidersMultipleChoice,
        },
        {
            ...chatcompletionsmessagesSection,
            quiz: chatcompletionsmessagesQuiz,
            multipleChoice: chatcompletionsmessagesMultipleChoice,
        },
        {
            ...tokenscontextwindowsSection,
            quiz: tokenscontextwindowsQuiz,
            multipleChoice: tokenscontextwindowsMultipleChoice,
        },
        {
            ...temperaturesamplingSection,
            quiz: temperaturesamplingQuiz,
            multipleChoice: temperaturesamplingMultipleChoice,
        },
        {
            ...streamingresponsesSection,
            quiz: streamingresponsesQuiz,
            multipleChoice: streamingresponsesMultipleChoice,
        },
        {
            ...errorhandlingretrySection,
            quiz: errorhandlingretryQuiz,
            multipleChoice: errorhandlingretryMultipleChoice,
        },
        {
            ...costtrackingoptimizationSection,
            quiz: costtrackingoptimizationQuiz,
            multipleChoice: costtrackingoptimizationMultipleChoice,
        },
        {
            ...prompttemplatesSection,
            quiz: prompttemplatesQuiz,
            multipleChoice: prompttemplatesMultipleChoice,
        },
        {
            ...outputparsingSection,
            quiz: outputparsingQuiz,
            multipleChoice: outputparsingMultipleChoice,
        },
        {
            ...observabilityloggingSection,
            quiz: observabilityloggingQuiz,
            multipleChoice: observabilityloggingMultipleChoice,
        },
        {
            ...cachingperformanceSection,
            quiz: cachingperformanceQuiz,
            multipleChoice: cachingperformanceMultipleChoice,
        },
        {
            ...localllmdeploymentSection,
            quiz: localllmdeploymentQuiz,
            multipleChoice: localllmdeploymentMultipleChoice,
        },
    ],
};
