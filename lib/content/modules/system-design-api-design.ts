/**
 * API Design & Management Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { restfulapidesignSection } from '../sections/system-design-api-design/restful-api-design';
import { apirequestresponsedesignSection } from '../sections/system-design-api-design/api-request-response-design';
import { apiauthenticationSection } from '../sections/system-design-api-design/api-authentication';
import { restapierrorhandlingSection } from '../sections/system-design-api-design/rest-api-error-handling';
import { graphqlschemadesignSection } from '../sections/system-design-api-design/graphql-schema-design';
import { graphqlperformanceSection } from '../sections/system-design-api-design/graphql-performance';
import { grpcservicedesignSection } from '../sections/system-design-api-design/grpc-service-design';
import { apigatewaypatternsSection } from '../sections/system-design-api-design/api-gateway-patterns';
import { apiratelimitingSection } from '../sections/system-design-api-design/api-rate-limiting';
import { apimonitoringSection } from '../sections/system-design-api-design/api-monitoring';
import { apidocumentationSection } from '../sections/system-design-api-design/api-documentation';
import { apiversioningSection } from '../sections/system-design-api-design/api-versioning';
import { webhookdesignSection } from '../sections/system-design-api-design/webhook-design';
import { apitestingSection } from '../sections/system-design-api-design/api-testing';
import { apigovernanceSection } from '../sections/system-design-api-design/api-governance';

// Import quizzes
import { restfulapidesignQuiz } from '../quizzes/system-design-api-design/restful-api-design';
import { apirequestresponsedesignQuiz } from '../quizzes/system-design-api-design/api-request-response-design';
import { apiauthenticationQuiz } from '../quizzes/system-design-api-design/api-authentication';
import { restapierrorhandlingQuiz } from '../quizzes/system-design-api-design/rest-api-error-handling';
import { graphqlschemadesignQuiz } from '../quizzes/system-design-api-design/graphql-schema-design';
import { graphqlperformanceQuiz } from '../quizzes/system-design-api-design/graphql-performance';
import { grpcservicedesignQuiz } from '../quizzes/system-design-api-design/grpc-service-design';
import { apigatewaypatternsQuiz } from '../quizzes/system-design-api-design/api-gateway-patterns';
import { apiratelimitingQuiz } from '../quizzes/system-design-api-design/api-rate-limiting';
import { apimonitoringQuiz } from '../quizzes/system-design-api-design/api-monitoring';
import { apidocumentationQuiz } from '../quizzes/system-design-api-design/api-documentation';
import { apiversioningQuiz } from '../quizzes/system-design-api-design/api-versioning';
import { webhookdesignQuiz } from '../quizzes/system-design-api-design/webhook-design';
import { apitestingQuiz } from '../quizzes/system-design-api-design/api-testing';
import { apigovernanceQuiz } from '../quizzes/system-design-api-design/api-governance';

// Import multiple choice
import { restfulapidesignMultipleChoice } from '../multiple-choice/system-design-api-design/restful-api-design';
import { apirequestresponsedesignMultipleChoice } from '../multiple-choice/system-design-api-design/api-request-response-design';
import { apiauthenticationMultipleChoice } from '../multiple-choice/system-design-api-design/api-authentication';
import { restapierrorhandlingMultipleChoice } from '../multiple-choice/system-design-api-design/rest-api-error-handling';
import { graphqlschemadesignMultipleChoice } from '../multiple-choice/system-design-api-design/graphql-schema-design';
import { graphqlperformanceMultipleChoice } from '../multiple-choice/system-design-api-design/graphql-performance';
import { grpcservicedesignMultipleChoice } from '../multiple-choice/system-design-api-design/grpc-service-design';
import { apigatewaypatternsMultipleChoice } from '../multiple-choice/system-design-api-design/api-gateway-patterns';
import { apiratelimitingMultipleChoice } from '../multiple-choice/system-design-api-design/api-rate-limiting';
import { apimonitoringMultipleChoice } from '../multiple-choice/system-design-api-design/api-monitoring';
import { apidocumentationMultipleChoice } from '../multiple-choice/system-design-api-design/api-documentation';
import { apiversioningMultipleChoice } from '../multiple-choice/system-design-api-design/api-versioning';
import { webhookdesignMultipleChoice } from '../multiple-choice/system-design-api-design/webhook-design';
import { apitestingMultipleChoice } from '../multiple-choice/system-design-api-design/api-testing';
import { apigovernanceMultipleChoice } from '../multiple-choice/system-design-api-design/api-governance';

export const systemDesignApiDesignModule: Module = {
  id: 'system-design-api-design',
  title: 'API Design & Management',
  description:
    'Master RESTful API design, GraphQL, gRPC, and API lifecycle management',
  category: 'undefined',
  difficulty: 'medium',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔌',
  keyTakeaways: [
    'RESTful API design uses HTTP standards effectively with proper verbs, status codes, and resource naming',
    'Pagination strategies: offset-based for simplicity, cursor-based for scale and consistency',
    'Field selection optimizes bandwidth without multiple endpoints or paradigm shifts',
    'Never embed unbounded nested collections; use separate paginated endpoints',
    'Structured error responses with codes, messages, and field-level details improve client experience',
    'Different authentication methods for different clients: OAuth for users, API keys for integrations, mTLS for internal services',
    'JWT tokens provide stateless authentication but require strategy for revocation',
    'Comprehensive error handling with proper status codes and detailed validation feedback is essential',
  ],
  learningObjectives: [
    'Design RESTful APIs following REST constraints and Richardson Maturity Model',
    'Choose appropriate pagination, filtering, and sorting strategies for different scales',
    'Implement comprehensive error handling with proper status codes and error structures',
    'Select and implement appropriate authentication methods for different client types',
    'Design secure token management with proper expiration, rotation, and revocation strategies',
    'Structure validation errors with field-level details for better developer experience',
    'Implement retry strategies and idempotency for reliable API operations',
    'Make informed architectural decisions considering scalability, security, and developer experience',
  ],
  sections: [
    {
      ...restfulapidesignSection,
      quiz: restfulapidesignQuiz,
      multipleChoice: restfulapidesignMultipleChoice,
    },
    {
      ...apirequestresponsedesignSection,
      quiz: apirequestresponsedesignQuiz,
      multipleChoice: apirequestresponsedesignMultipleChoice,
    },
    {
      ...apiauthenticationSection,
      quiz: apiauthenticationQuiz,
      multipleChoice: apiauthenticationMultipleChoice,
    },
    {
      ...restapierrorhandlingSection,
      quiz: restapierrorhandlingQuiz,
      multipleChoice: restapierrorhandlingMultipleChoice,
    },
    {
      ...graphqlschemadesignSection,
      quiz: graphqlschemadesignQuiz,
      multipleChoice: graphqlschemadesignMultipleChoice,
    },
    {
      ...graphqlperformanceSection,
      quiz: graphqlperformanceQuiz,
      multipleChoice: graphqlperformanceMultipleChoice,
    },
    {
      ...grpcservicedesignSection,
      quiz: grpcservicedesignQuiz,
      multipleChoice: grpcservicedesignMultipleChoice,
    },
    {
      ...apigatewaypatternsSection,
      quiz: apigatewaypatternsQuiz,
      multipleChoice: apigatewaypatternsMultipleChoice,
    },
    {
      ...apiratelimitingSection,
      quiz: apiratelimitingQuiz,
      multipleChoice: apiratelimitingMultipleChoice,
    },
    {
      ...apimonitoringSection,
      quiz: apimonitoringQuiz,
      multipleChoice: apimonitoringMultipleChoice,
    },
    {
      ...apidocumentationSection,
      quiz: apidocumentationQuiz,
      multipleChoice: apidocumentationMultipleChoice,
    },
    {
      ...apiversioningSection,
      quiz: apiversioningQuiz,
      multipleChoice: apiversioningMultipleChoice,
    },
    {
      ...webhookdesignSection,
      quiz: webhookdesignQuiz,
      multipleChoice: webhookdesignMultipleChoice,
    },
    {
      ...apitestingSection,
      quiz: apitestingQuiz,
      multipleChoice: apitestingMultipleChoice,
    },
    {
      ...apigovernanceSection,
      quiz: apigovernanceQuiz,
      multipleChoice: apigovernanceMultipleChoice,
    },
  ],
};
