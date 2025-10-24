/**
 * Module: Production LLM Applications
 * Module 12 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports - using content exports
import { productionArchitecturePatternsContent } from '../sections/production-llm-applications/production-architecture-patterns';
import { apiDesignForLlmAppsContent } from '../sections/production-llm-applications/api-design-for-llm-apps';
import { asyncConcurrencyContent } from '../sections/production-llm-applications/async-concurrency';
import { queueSystemsBackgroundJobsContent } from '../sections/production-llm-applications/queue-systems-background-jobs';
import { cachingStrategiesContent } from '../sections/production-llm-applications/caching-strategies';
import { rateLimitingThrottlingContent } from '../sections/production-llm-applications/rate-limiting-throttling';
import { errorHandlingResilienceContent } from '../sections/production-llm-applications/error-handling-resilience';
import { monitoringObservabilityContent } from '../sections/production-llm-applications/monitoring-observability';
import { databaseIntegrationContent } from '../sections/production-llm-applications/database-integration';
import { authenticationAuthorizationContent } from '../sections/production-llm-applications/authentication-authorization';
import { testingLlmApplicationsContent } from '../sections/production-llm-applications/testing-llm-applications';
import { deploymentStrategiesContent } from '../sections/production-llm-applications/deployment-strategies';
import { costManagementContent } from '../sections/production-llm-applications/cost-management';
import { buildingSaasLlmProductContent } from '../sections/production-llm-applications/building-saas-llm-product';

// Quiz imports
import { productionArchitecturePatternsQuiz } from '../quizzes/production-llm-applications/production-architecture-patterns';
import { apiDesignForLlmAppsQuiz } from '../quizzes/production-llm-applications/api-design-for-llm-apps';
import { asyncConcurrencyQuiz } from '../quizzes/production-llm-applications/async-concurrency';
import { queueSystemsBackgroundJobsQuiz } from '../quizzes/production-llm-applications/queue-systems-background-jobs';
import { cachingStrategiesQuiz } from '../quizzes/production-llm-applications/caching-strategies';
import { rateLimitingThrottlingQuiz } from '../quizzes/production-llm-applications/rate-limiting-throttling';
import { errorHandlingResilienceQuiz } from '../quizzes/production-llm-applications/error-handling-resilience';
import { monitoringObservabilityQuiz } from '../quizzes/production-llm-applications/monitoring-observability';
import { databaseIntegrationQuiz } from '../quizzes/production-llm-applications/database-integration';
import { authenticationAuthorizationQuiz } from '../quizzes/production-llm-applications/authentication-authorization';
import { testingLlmApplicationsQuiz } from '../quizzes/production-llm-applications/testing-llm-applications';
import { deploymentStrategiesQuiz } from '../quizzes/production-llm-applications/deployment-strategies';
import { costManagementQuiz } from '../quizzes/production-llm-applications/cost-management';
import { buildingSaasLlmProductQuiz } from '../quizzes/production-llm-applications/building-saas-llm-product';

// Multiple choice imports
import { productionArchitecturePatternsMultipleChoice } from '../multiple-choice/production-llm-applications/production-architecture-patterns';
import { apiDesignForLlmAppsMultipleChoice } from '../multiple-choice/production-llm-applications/api-design-for-llm-apps';
import { asyncConcurrencyMultipleChoice } from '../multiple-choice/production-llm-applications/async-concurrency';
import { queueSystemsBackgroundJobsMultipleChoice } from '../multiple-choice/production-llm-applications/queue-systems-background-jobs';
import { cachingStrategiesMultipleChoice } from '../multiple-choice/production-llm-applications/caching-strategies';
import { rateLimitingThrottlingMultipleChoice } from '../multiple-choice/production-llm-applications/rate-limiting-throttling';
import { errorHandlingResilienceMultipleChoice } from '../multiple-choice/production-llm-applications/error-handling-resilience';
import { monitoringObservabilityMultipleChoice } from '../multiple-choice/production-llm-applications/monitoring-observability';
import { databaseIntegrationMultipleChoice } from '../multiple-choice/production-llm-applications/database-integration';
import { authenticationAuthorizationMultipleChoice } from '../multiple-choice/production-llm-applications/authentication-authorization';
import { testingLlmApplicationsMultipleChoice } from '../multiple-choice/production-llm-applications/testing-llm-applications';
import { deploymentStrategiesMultipleChoice } from '../multiple-choice/production-llm-applications/deployment-strategies';
import { costManagementMultipleChoice } from '../multiple-choice/production-llm-applications/cost-management';
import { buildingSaasLlmProductMultipleChoice } from '../multiple-choice/production-llm-applications/building-saas-llm-product';

export const productionLlmApplicationsModule: Module = {
  id: 'production-llm-applications',
  title: 'Production LLM Applications',
  description:
    'Master building production-ready LLM applications that scale to thousands of users. Learn architecture patterns, API design, async processing, caching, monitoring, and deployment strategies used by successful AI companies.',
  icon: '🚀',
  keyTakeaways: [
    'Design scalable microservices and event-driven architectures for LLM apps',
    'Build production APIs with streaming, versioning, and proper error handling',
    'Implement async/await patterns for 10x throughput improvements',
    'Master queue systems (Celery, RQ) for background job processing',
    'Achieve 70-90% cost reduction through aggressive caching strategies',
    'Implement tiered rate limiting and cost-based budget management',
    'Build resilient systems with circuit breakers and graceful degradation',
    'Deploy comprehensive monitoring with metrics, logging, and distributed tracing',
    'Scale databases with connection pooling, indexes, and vector search',
    'Secure applications with authentication, authorization, and multi-tenancy',
    'Test LLM applications efficiently with mocks and limited real API usage',
    'Deploy with Docker, Kubernetes, and CI/CD pipelines for zero-downtime',
    'Track and optimize costs with granular per-user, per-model analytics',
    'Build complete SaaS products with subscriptions, billing, and admin dashboards',
  ],
  sections: [
    {
      id: 'production-architecture-patterns',
      title: 'Production Architecture Patterns',
      content: productionArchitecturePatternsContent,
      quiz: productionArchitecturePatternsQuiz,
      multipleChoice: productionArchitecturePatternsMultipleChoice,
    },
    {
      id: 'api-design-for-llm-apps',
      title: 'API Design for LLM Apps',
      content: apiDesignForLlmAppsContent,
      quiz: apiDesignForLlmAppsQuiz,
      multipleChoice: apiDesignForLlmAppsMultipleChoice,
    },
    {
      id: 'async-concurrency',
      title: 'Async & Concurrency',
      content: asyncConcurrencyContent,
      quiz: asyncConcurrencyQuiz,
      multipleChoice: asyncConcurrencyMultipleChoice,
    },
    {
      id: 'queue-systems-background-jobs',
      title: 'Queue Systems & Background Jobs',
      content: queueSystemsBackgroundJobsContent,
      quiz: queueSystemsBackgroundJobsQuiz,
      multipleChoice: queueSystemsBackgroundJobsMultipleChoice,
    },
    {
      id: 'caching-strategies',
      title: 'Caching Strategies',
      content: cachingStrategiesContent,
      quiz: cachingStrategiesQuiz,
      multipleChoice: cachingStrategiesMultipleChoice,
    },
    {
      id: 'rate-limiting-throttling',
      title: 'Rate Limiting & Throttling',
      content: rateLimitingThrottlingContent,
      quiz: rateLimitingThrottlingQuiz,
      multipleChoice: rateLimitingThrottlingMultipleChoice,
    },
    {
      id: 'error-handling-resilience',
      title: 'Error Handling & Resilience',
      content: errorHandlingResilienceContent,
      quiz: errorHandlingResilienceQuiz,
      multipleChoice: errorHandlingResilienceMultipleChoice,
    },
    {
      id: 'monitoring-observability',
      title: 'Monitoring & Observability',
      content: monitoringObservabilityContent,
      quiz: monitoringObservabilityQuiz,
      multipleChoice: monitoringObservabilityMultipleChoice,
    },
    {
      id: 'database-integration',
      title: 'Database Integration',
      content: databaseIntegrationContent,
      quiz: databaseIntegrationQuiz,
      multipleChoice: databaseIntegrationMultipleChoice,
    },
    {
      id: 'authentication-authorization',
      title: 'Authentication & Authorization',
      content: authenticationAuthorizationContent,
      quiz: authenticationAuthorizationQuiz,
      multipleChoice: authenticationAuthorizationMultipleChoice,
    },
    {
      id: 'testing-llm-applications',
      title: 'Testing LLM Applications',
      content: testingLlmApplicationsContent,
      quiz: testingLlmApplicationsQuiz,
      multipleChoice: testingLlmApplicationsMultipleChoice,
    },
    {
      id: 'deployment-strategies',
      title: 'Deployment Strategies',
      content: deploymentStrategiesContent,
      quiz: deploymentStrategiesQuiz,
      multipleChoice: deploymentStrategiesMultipleChoice,
    },
    {
      id: 'cost-management',
      title: 'Cost Management',
      content: costManagementContent,
      quiz: costManagementQuiz,
      multipleChoice: costManagementMultipleChoice,
    },
    {
      id: 'building-saas-llm-product',
      title: 'Building a SaaS LLM Product',
      content: buildingSaasLlmProductContent,
      quiz: buildingSaasLlmProductQuiz,
      multipleChoice: buildingSaasLlmProductMultipleChoice,
    },
  ],
};
