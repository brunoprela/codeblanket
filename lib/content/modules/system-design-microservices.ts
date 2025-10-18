/**
 * Microservices Architecture Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { microservicesvsmonolithSection } from '../sections/system-design-microservices/microservices-vs-monolith';
import { servicedecompositionSection } from '../sections/system-design-microservices/service-decomposition';
import { interservicecommunicationSection } from '../sections/system-design-microservices/inter-service-communication';
import { servicediscoverySection } from '../sections/system-design-microservices/service-discovery';
import { apigatewaySection } from '../sections/system-design-microservices/api-gateway';
import { distributedtransactionssagaSection } from '../sections/system-design-microservices/distributed-transactions-saga';
import { datamanagementmicroservicesSection } from '../sections/system-design-microservices/data-management-microservices';
import { circuitbreakerSection } from '../sections/system-design-microservices/circuit-breaker';
import { servicemeshSection } from '../sections/system-design-microservices/service-mesh';
import { microservicestestingSection } from '../sections/system-design-microservices/microservices-testing';
import { microservicesdeploymentSection } from '../sections/system-design-microservices/microservices-deployment';
import { microservicessecuritySection } from '../sections/system-design-microservices/microservices-security';
import { microservicesmonitoringSection } from '../sections/system-design-microservices/microservices-monitoring';
import { eventdrivenmicroservicesSection } from '../sections/system-design-microservices/event-driven-microservices';
import { microservicesantipatternsSection } from '../sections/system-design-microservices/microservices-anti-patterns';

// Import quizzes
import { microservicesvsmonolithQuiz } from '../quizzes/system-design-microservices/microservices-vs-monolith';
import { servicedecompositionQuiz } from '../quizzes/system-design-microservices/service-decomposition';
import { interservicecommunicationQuiz } from '../quizzes/system-design-microservices/inter-service-communication';
import { servicediscoveryQuiz } from '../quizzes/system-design-microservices/service-discovery';
import { apigatewayQuiz } from '../quizzes/system-design-microservices/api-gateway';
import { distributedtransactionssagaQuiz } from '../quizzes/system-design-microservices/distributed-transactions-saga';
import { datamanagementmicroservicesQuiz } from '../quizzes/system-design-microservices/data-management-microservices';
import { circuitbreakerQuiz } from '../quizzes/system-design-microservices/circuit-breaker';
import { servicemeshQuiz } from '../quizzes/system-design-microservices/service-mesh';
import { microservicestestingQuiz } from '../quizzes/system-design-microservices/microservices-testing';
import { microservicesdeploymentQuiz } from '../quizzes/system-design-microservices/microservices-deployment';
import { microservicessecurityQuiz } from '../quizzes/system-design-microservices/microservices-security';
import { microservicesmonitoringQuiz } from '../quizzes/system-design-microservices/microservices-monitoring';
import { eventdrivenmicroservicesQuiz } from '../quizzes/system-design-microservices/event-driven-microservices';
import { microservicesantipatternsQuiz } from '../quizzes/system-design-microservices/microservices-anti-patterns';

// Import multiple choice
import { microservicesvsmonolithMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-vs-monolith';
import { servicedecompositionMultipleChoice } from '../multiple-choice/system-design-microservices/service-decomposition';
import { interservicecommunicationMultipleChoice } from '../multiple-choice/system-design-microservices/inter-service-communication';
import { servicediscoveryMultipleChoice } from '../multiple-choice/system-design-microservices/service-discovery';
import { apigatewayMultipleChoice } from '../multiple-choice/system-design-microservices/api-gateway';
import { distributedtransactionssagaMultipleChoice } from '../multiple-choice/system-design-microservices/distributed-transactions-saga';
import { datamanagementmicroservicesMultipleChoice } from '../multiple-choice/system-design-microservices/data-management-microservices';
import { circuitbreakerMultipleChoice } from '../multiple-choice/system-design-microservices/circuit-breaker';
import { servicemeshMultipleChoice } from '../multiple-choice/system-design-microservices/service-mesh';
import { microservicestestingMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-testing';
import { microservicesdeploymentMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-deployment';
import { microservicessecurityMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-security';
import { microservicesmonitoringMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-monitoring';
import { eventdrivenmicroservicesMultipleChoice } from '../multiple-choice/system-design-microservices/event-driven-microservices';
import { microservicesantipatternsMultipleChoice } from '../multiple-choice/system-design-microservices/microservices-anti-patterns';

export const systemDesignMicroservicesModule: Module = {
  id: 'system-design-microservices',
  title: 'Microservices Architecture',
  description:
    'Master microservices patterns, decomposition strategies, and distributed system challenges',
  category: 'System Design',
  difficulty: 'Advanced',
  estimatedTime: '4-5 hours',
  prerequisites: [],
  icon: '🔬',
  keyTakeaways: [
    'Microservices vs Monolith: Choose based on team size, scale, and operational maturity',
    'Service Decomposition: Follow domain boundaries (DDD), not technical layers',
    'Inter-Service Communication: Balance sync (simple) vs async (resilient)',
    'Data Management: Database per service to enable true independence',
    'Circuit Breakers: Prevent cascading failures in distributed systems',
    'Service Mesh: Provides observability, security, and traffic management',
    'Saga Pattern: Handle distributed transactions with eventual consistency',
    'Start Simple: Monolith first, extract services strategically using Strangler Fig',
  ],
  learningObjectives: [
    'Understand trade-offs between monolithic and microservices architectures',
    'Master service decomposition strategies using business capabilities and DDD',
    'Design effective inter-service communication patterns',
    'Implement resilience patterns (circuit breakers, retries, timeouts)',
    'Handle data consistency in distributed systems using Saga pattern',
    'Build observable microservices with distributed tracing',
    'Design for failure with fault isolation and graceful degradation',
    'Apply microservices patterns to real-world system design problems',
  ],
  sections: [
    {
      ...microservicesvsmonolithSection,
      quiz: microservicesvsmonolithQuiz,
      multipleChoice: microservicesvsmonolithMultipleChoice,
    },
    {
      ...servicedecompositionSection,
      quiz: servicedecompositionQuiz,
      multipleChoice: servicedecompositionMultipleChoice,
    },
    {
      ...interservicecommunicationSection,
      quiz: interservicecommunicationQuiz,
      multipleChoice: interservicecommunicationMultipleChoice,
    },
    {
      ...servicediscoverySection,
      quiz: servicediscoveryQuiz,
      multipleChoice: servicediscoveryMultipleChoice,
    },
    {
      ...apigatewaySection,
      quiz: apigatewayQuiz,
      multipleChoice: apigatewayMultipleChoice,
    },
    {
      ...distributedtransactionssagaSection,
      quiz: distributedtransactionssagaQuiz,
      multipleChoice: distributedtransactionssagaMultipleChoice,
    },
    {
      ...datamanagementmicroservicesSection,
      quiz: datamanagementmicroservicesQuiz,
      multipleChoice: datamanagementmicroservicesMultipleChoice,
    },
    {
      ...circuitbreakerSection,
      quiz: circuitbreakerQuiz,
      multipleChoice: circuitbreakerMultipleChoice,
    },
    {
      ...servicemeshSection,
      quiz: servicemeshQuiz,
      multipleChoice: servicemeshMultipleChoice,
    },
    {
      ...microservicestestingSection,
      quiz: microservicestestingQuiz,
      multipleChoice: microservicestestingMultipleChoice,
    },
    {
      ...microservicesdeploymentSection,
      quiz: microservicesdeploymentQuiz,
      multipleChoice: microservicesdeploymentMultipleChoice,
    },
    {
      ...microservicessecuritySection,
      quiz: microservicessecurityQuiz,
      multipleChoice: microservicessecurityMultipleChoice,
    },
    {
      ...microservicesmonitoringSection,
      quiz: microservicesmonitoringQuiz,
      multipleChoice: microservicesmonitoringMultipleChoice,
    },
    {
      ...eventdrivenmicroservicesSection,
      quiz: eventdrivenmicroservicesQuiz,
      multipleChoice: eventdrivenmicroservicesMultipleChoice,
    },
    {
      ...microservicesantipatternsSection,
      quiz: microservicesantipatternsQuiz,
      multipleChoice: microservicesantipatternsMultipleChoice,
    },
  ],
};
