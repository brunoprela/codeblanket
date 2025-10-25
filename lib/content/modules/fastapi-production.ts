/**
 * Module: FastAPI Production Mastery
 *
 * Comprehensive production-ready FastAPI curriculum
 * 17 sections covering architecture to deployment
 */

import { Module } from '../../types';

// Section imports
import { fastapiArchitecturePhilosophy } from '../sections/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydantic } from '../sections/fastapi-production/request-response-models-pydantic';
import { pathOperationsRouting } from '../sections/fastapi-production/path-operations-routing';
import { dependencyInjectionSystem } from '../sections/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemy } from '../sections/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2 } from '../sections/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissions } from '../sections/fastapi-production/authorization-permissions';
import { backgroundTasks } from '../sections/fastapi-production/background-tasks';
import { websocketsRealtime } from '../sections/fastapi-production/websockets-realtime';
import { fileUploadsStreaming } from '../sections/fastapi-production/file-uploads-streaming';
import { errorHandlingValidation } from '../sections/fastapi-production/error-handling-validation';
import { middlewareCors } from '../sections/fastapi-production/middleware-cors';
import { apiDocumentation } from '../sections/fastapi-production/api-documentation';
import { testingFastapi } from '../sections/fastapi-production/testing-fastapi';
import { asyncPatterns } from '../sections/fastapi-production/async-patterns';
import { productionDeployment } from '../sections/fastapi-production/production-deployment';
import { bestPracticesPatterns } from '../sections/fastapi-production/best-practices-patterns';

// Quiz imports
import { fastapiArchitecturePhilosophyQuiz } from '../quizzes/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydanticQuiz } from '../quizzes/fastapi-production/request-response-models-pydantic';
import { pathOperationsRoutingQuiz } from '../quizzes/fastapi-production/path-operations-routing';
import { dependencyInjectionSystemQuiz } from '../quizzes/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemyQuiz } from '../quizzes/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2Quiz } from '../quizzes/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissionsQuiz } from '../quizzes/fastapi-production/authorization-permissions';
import { backgroundTasksQuiz } from '../quizzes/fastapi-production/background-tasks';
import { websocketsRealtimeQuiz } from '../quizzes/fastapi-production/websockets-realtime';
import { fileUploadsStreamingQuiz } from '../quizzes/fastapi-production/file-uploads-streaming';
import { errorHandlingValidationQuiz } from '../quizzes/fastapi-production/error-handling-validation';
import { middlewareCorsQuiz } from '../quizzes/fastapi-production/middleware-cors';
import { apiDocumentationQuiz } from '../quizzes/fastapi-production/api-documentation';
import { testingFastapiQuiz } from '../quizzes/fastapi-production/testing-fastapi';
import { asyncPatternsQuiz } from '../quizzes/fastapi-production/async-patterns';
import { productionDeploymentQuiz } from '../quizzes/fastapi-production/production-deployment';
import { bestPracticesPatternsQuiz } from '../quizzes/fastapi-production/best-practices-patterns';

// Multiple choice imports
import { fastapiArchitecturePhilosophyMultipleChoice } from '../multiple-choice/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydanticMultipleChoice } from '../multiple-choice/fastapi-production/request-response-models-pydantic';
import { pathOperationsRoutingMultipleChoice } from '../multiple-choice/fastapi-production/path-operations-routing';
import { dependencyInjectionSystemMultipleChoice } from '../multiple-choice/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemyMultipleChoice } from '../multiple-choice/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2MultipleChoice } from '../multiple-choice/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissionsMultipleChoice } from '../multiple-choice/fastapi-production/authorization-permissions';
import { backgroundTasksMultipleChoice } from '../multiple-choice/fastapi-production/background-tasks';
import { websocketsRealtimeMultipleChoice } from '../multiple-choice/fastapi-production/websockets-realtime';
import { fileUploadsStreamingMultipleChoice } from '../multiple-choice/fastapi-production/file-uploads-streaming';
import { errorHandlingValidationMultipleChoice } from '../multiple-choice/fastapi-production/error-handling-validation';
import { middlewareCorsMultipleChoice } from '../multiple-choice/fastapi-production/middleware-cors';
import { apiDocumentationMultipleChoice } from '../multiple-choice/fastapi-production/api-documentation';
import { testingFastapiMultipleChoice } from '../multiple-choice/fastapi-production/testing-fastapi';
import { asyncPatternsMultipleChoice } from '../multiple-choice/fastapi-production/async-patterns';
import { productionDeploymentMultipleChoice } from '../multiple-choice/fastapi-production/production-deployment';
import { bestPracticesPatternsMultipleChoice } from '../multiple-choice/fastapi-production/best-practices-patterns';

export const fastapiProductionModule: Module = {
  id: 'fastapi-production',
  title: 'FastAPI Production Mastery',
  description:
    'Master production-ready API development with FastAPI: from architecture and authentication to deployment and best practices. Build scalable, secure, high-performance APIs.',
  icon: '🚀',
  difficulty: 'Advanced',
  estimatedTime: '40 hours',

  sections: [
    {
      ...fastapiArchitecturePhilosophy,
      quiz: fastapiArchitecturePhilosophyQuiz,
      multipleChoice: fastapiArchitecturePhilosophyMultipleChoice,
    },
    {
      ...requestResponseModelsPydantic,
      quiz: requestResponseModelsPydanticQuiz,
      multipleChoice: requestResponseModelsPydanticMultipleChoice,
    },
    {
      ...pathOperationsRouting,
      quiz: pathOperationsRoutingQuiz,
      multipleChoice: pathOperationsRoutingMultipleChoice,
    },
    {
      ...dependencyInjectionSystem,
      quiz: dependencyInjectionSystemQuiz,
      multipleChoice: dependencyInjectionSystemMultipleChoice,
    },
    {
      ...databaseIntegrationSqlalchemy,
      quiz: databaseIntegrationSqlalchemyQuiz,
      multipleChoice: databaseIntegrationSqlalchemyMultipleChoice,
    },
    {
      ...authenticationJwtOauth2,
      quiz: authenticationJwtOauth2Quiz,
      multipleChoice: authenticationJwtOauth2MultipleChoice,
    },
    {
      ...authorizationPermissions,
      quiz: authorizationPermissionsQuiz,
      multipleChoice: authorizationPermissionsMultipleChoice,
    },
    {
      ...backgroundTasks,
      quiz: backgroundTasksQuiz,
      multipleChoice: backgroundTasksMultipleChoice,
    },
    {
      ...websocketsRealtime,
      quiz: websocketsRealtimeQuiz,
      multipleChoice: websocketsRealtimeMultipleChoice,
    },
    {
      ...fileUploadsStreaming,
      quiz: fileUploadsStreamingQuiz,
      multipleChoice: fileUploadsStreamingMultipleChoice,
    },
    {
      ...errorHandlingValidation,
      quiz: errorHandlingValidationQuiz,
      multipleChoice: errorHandlingValidationMultipleChoice,
    },
    {
      ...middlewareCors,
      quiz: middlewareCorsQuiz,
      multipleChoice: middlewareCorsMultipleChoice,
    },
    {
      ...apiDocumentation,
      quiz: apiDocumentationQuiz,
      multipleChoice: apiDocumentationMultipleChoice,
    },
    {
      ...testingFastapi,
      quiz: testingFastapiQuiz,
      multipleChoice: testingFastapiMultipleChoice,
    },
    {
      ...asyncPatterns,
      quiz: asyncPatternsQuiz,
      multipleChoice: asyncPatternsMultipleChoice,
    },
    {
      ...productionDeployment,
      quiz: productionDeploymentQuiz,
      multipleChoice: productionDeploymentMultipleChoice,
    },
    {
      ...bestPracticesPatterns,
      quiz: bestPracticesPatternsQuiz,
      multipleChoice: bestPracticesPatternsMultipleChoice,
    },
  ],

  keyTakeaways: [
    'Master FastAPI architecture and dependency injection for clean, testable code',
    'Implement secure authentication with JWT and OAuth2 flows',
    'Build role-based authorization and fine-grained permissions',
    'Integrate SQLAlchemy with async support for high-performance database access',
    'Create production-ready APIs with error handling, logging, and monitoring',
    'Deploy FastAPI applications with Docker, Kubernetes, and zero-downtime strategies',
    'Test FastAPI applications comprehensively with pytest and dependency mocking',
    'Optimize performance with async patterns, caching, and connection pooling',
  ],

  learningObjectives: [
    'Understand FastAPI architecture: ASGI, Pydantic, dependency injection',
    'Design RESTful APIs with proper HTTP methods, status codes, and routing',
    'Implement JWT authentication and OAuth2 password flow',
    'Build RBAC and PBAC authorization systems',
    'Integrate PostgreSQL with SQLAlchemy ORM and Alembic migrations',
    'Handle file uploads, WebSockets, and background tasks',
    'Configure middleware for CORS, logging, and rate limiting',
    'Deploy to production with Uvicorn, Gunicorn, and Docker',
    'Monitor APIs with Prometheus, Grafana, and Sentry',
    'Apply security best practices and OWASP guidelines',
  ],

  prerequisites: [
    'Python fundamentals (functions, classes, decorators)',
    'Understanding of web APIs and HTTP',
    'Basic SQL and database concepts',
    'Familiarity with async/await (helpful but not required)',
    'Command line and Git basics',
  ],

  practicalProjects: [
    {
      title: 'Multi-Tenant SaaS API',
      description:
        'Build a complete multi-tenant API with authentication, authorization, and data isolation',
      difficulty: 'Advanced',
      estimatedTime: '10 hours',
    },
    {
      title: 'Real-Time Chat Application',
      description:
        'Implement WebSocket-based chat with FastAPI, including authentication and message persistence',
      difficulty: 'Intermediate',
      estimatedTime: '6 hours',
    },
    {
      title: 'E-Commerce Backend',
      description:
        'Create a production-ready e-commerce API with products, orders, payments, and admin dashboard',
      difficulty: 'Advanced',
      estimatedTime: '15 hours',
    },
    {
      title: 'File Processing Service',
      description:
        'Build an API for uploading, processing, and serving files with background task processing',
      difficulty: 'Intermediate',
      estimatedTime: '5 hours',
    },
  ],
};
