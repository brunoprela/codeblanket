/**
 * Observability & Resilience Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { observabilityFundamentalsSection } from '../sections/observability-resilience/observability-fundamentals';
import { loggingBestPracticesSection } from '../sections/observability-resilience/logging-best-practices';
import { metricsMonitoringSection } from '../sections/observability-resilience/metrics-monitoring';
import { distributedTracingSection } from '../sections/observability-resilience/distributed-tracing';
import { apmSection } from '../sections/observability-resilience/apm';
import { alertingStrategiesSection } from '../sections/observability-resilience/alerting-strategies';
import { slisSection } from '../sections/observability-resilience/slis-slos-slas';
import { circuitBreakerBulkheadSection } from '../sections/observability-resilience/circuit-breaker-bulkhead';
import { retryLogicSection } from '../sections/observability-resilience/retry-logic-exponential-backoff';
import { chaosEngineeringSection } from '../sections/observability-resilience/chaos-engineering';
import { incidentManagementSection } from '../sections/observability-resilience/incident-management';
import { healthChecksSection } from '../sections/observability-resilience/health-checks-readiness-probes';

// Import quizzes
import { observabilityFundamentalsQuiz } from '../quizzes/observability-resilience/observability-fundamentals-quiz';
import { loggingBestPracticesQuiz } from '../quizzes/observability-resilience/logging-best-practices-quiz';
import { metricsMonitoringQuiz } from '../quizzes/observability-resilience/metrics-monitoring-quiz';
import { distributedTracingQuiz } from '../quizzes/observability-resilience/distributed-tracing-quiz';
import { apmQuiz } from '../quizzes/observability-resilience/apm-quiz';
import { alertingStrategiesQuiz } from '../quizzes/observability-resilience/alerting-strategies-quiz';
import { slisQuiz } from '../quizzes/observability-resilience/slis-slos-slas-quiz';
import { circuitBreakerBulkheadQuiz } from '../quizzes/observability-resilience/circuit-breaker-bulkhead-quiz';
import { retryLogicQuiz } from '../quizzes/observability-resilience/retry-logic-exponential-backoff-quiz';
import { chaosEngineeringQuiz } from '../quizzes/observability-resilience/chaos-engineering-quiz';
import { incidentManagementQuiz } from '../quizzes/observability-resilience/incident-management-quiz';
import { healthChecksQuiz } from '../quizzes/observability-resilience/health-checks-readiness-probes-quiz';

// Import multiple choice
import { observabilityFundamentalsMultipleChoice } from '../multiple-choice/observability-resilience/observability-fundamentals-mc';
import { loggingBestPracticesMultipleChoice } from '../multiple-choice/observability-resilience/logging-best-practices-mc';
import { metricsMonitoringMultipleChoice } from '../multiple-choice/observability-resilience/metrics-monitoring-mc';
import { distributedTracingMultipleChoice } from '../multiple-choice/observability-resilience/distributed-tracing-mc';
import { apmMultipleChoice } from '../multiple-choice/observability-resilience/apm-mc';
import { alertingStrategiesMultipleChoice } from '../multiple-choice/observability-resilience/alerting-strategies-mc';
import { slisMultipleChoice } from '../multiple-choice/observability-resilience/slis-slos-slas-mc';
import { circuitBreakerBulkheadMultipleChoice } from '../multiple-choice/observability-resilience/circuit-breaker-bulkhead-mc';
import { retryLogicMultipleChoice } from '../multiple-choice/observability-resilience/retry-logic-exponential-backoff-mc';
import { chaosEngineeringMultipleChoice } from '../multiple-choice/observability-resilience/chaos-engineering-mc';
import { incidentManagementMultipleChoice } from '../multiple-choice/observability-resilience/incident-management-mc';
import { healthChecksMultipleChoice } from '../multiple-choice/observability-resilience/health-checks-readiness-probes-mc';

export const observabilityResilienceModule: Module = {
  id: 'observability-resilience',
  title: 'Observability & Resilience',
  description:
    'Master monitoring, logging, tracing, and building resilient systems that can withstand failures',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📊',
  keyTakeaways: [
    'Observability has three pillars: logs (discrete events), metrics (aggregated numbers), and traces (request journeys)',
    'Alert on symptoms (user impact) not causes (CPU usage), using SLO-based error budgets to drive decisions',
    'Circuit breakers fail fast to prevent cascading failures, while bulkheads isolate failures to limit blast radius',
    'Retry with exponential backoff and jitter to handle transient failures without overwhelming services',
    'Distributed tracing tracks requests across services, essential for debugging microservices architectures',
    'SLIs measure service behavior, SLOs set targets, and SLAs are contracts with consequences',
    'Chaos engineering proactively injects failures to build confidence in system resilience',
    'Effective incident management requires clear roles, blameless post-mortems, and action items tracked to completion',
  ],
  learningObjectives: [
    'Understand the three pillars of observability and when to use each',
    'Implement structured logging with correlation IDs for distributed systems',
    'Design metrics systems with appropriate cardinality and alerting strategies',
    'Set up distributed tracing to debug performance issues across services',
    'Use APM tools to identify bottlenecks and optimize application performance',
    'Create SLI/SLO/SLA frameworks to measure and maintain reliability',
    'Implement resilience patterns like circuit breakers, bulkheads, and retry logic',
    'Conduct chaos engineering experiments to validate system resilience',
    'Manage incidents effectively with proper communication and post-mortems',
    'Configure health checks and readiness probes for self-healing systems',
  ],
  sections: [
    {
      ...observabilityFundamentalsSection,
      quiz: observabilityFundamentalsQuiz,
      multipleChoice: observabilityFundamentalsMultipleChoice,
    },
    {
      ...loggingBestPracticesSection,
      quiz: loggingBestPracticesQuiz,
      multipleChoice: loggingBestPracticesMultipleChoice,
    },
    {
      ...metricsMonitoringSection,
      quiz: metricsMonitoringQuiz,
      multipleChoice: metricsMonitoringMultipleChoice,
    },
    {
      ...distributedTracingSection,
      quiz: distributedTracingQuiz,
      multipleChoice: distributedTracingMultipleChoice,
    },
    {
      ...apmSection,
      quiz: apmQuiz,
      multipleChoice: apmMultipleChoice,
    },
    {
      ...alertingStrategiesSection,
      quiz: alertingStrategiesQuiz,
      multipleChoice: alertingStrategiesMultipleChoice,
    },
    {
      ...slisSection,
      quiz: slisQuiz,
      multipleChoice: slisMultipleChoice,
    },
    {
      ...circuitBreakerBulkheadSection,
      quiz: circuitBreakerBulkheadQuiz,
      multipleChoice: circuitBreakerBulkheadMultipleChoice,
    },
    {
      ...retryLogicSection,
      quiz: retryLogicQuiz,
      multipleChoice: retryLogicMultipleChoice,
    },
    {
      ...chaosEngineeringSection,
      quiz: chaosEngineeringQuiz,
      multipleChoice: chaosEngineeringMultipleChoice,
    },
    {
      ...incidentManagementSection,
      quiz: incidentManagementQuiz,
      multipleChoice: incidentManagementMultipleChoice,
    },
    {
      ...healthChecksSection,
      quiz: healthChecksQuiz,
      multipleChoice: healthChecksMultipleChoice,
    },
  ],
};
