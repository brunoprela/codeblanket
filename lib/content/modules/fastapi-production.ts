/**
 * Module: FastAPI Production Mastery
 *
 * Comprehensive production-ready FastAPI curriculum
 * 17 sections covering architecture to deployment
 */

import { Module } from '../types';

// Section imports
import { fastapiArchitecturePhilosophy } from './sections/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydantic } from './sections/fastapi-production/request-response-models-pydantic';
import { pathOperationsRouting } from './sections/fastapi-production/path-operations-routing';
import { dependencyInjectionSystem } from './sections/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemy } from './sections/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2 } from './sections/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissions } from './sections/fastapi-production/authorization-permissions';
import { backgroundTasks } from './sections/fastapi-production/background-tasks';
import { websocketsRealtime } from './sections/fastapi-production/websockets-realtime';
import { fileUploadsStreaming } from './sections/fastapi-production/file-uploads-streaming';
import { errorHandlingValidation } from './sections/fastapi-production/error-handling-validation';
import { middlewareCors } from './sections/fastapi-production/middleware-cors';
import { apiDocumentation } from './sections/fastapi-production/api-documentation';
import { testingFastapi } from './sections/fastapi-production/testing-fastapi';
import { asyncPatterns } from './sections/fastapi-production/async-patterns';
import { productionDeployment } from './sections/fastapi-production/production-deployment';
import { bestPracticesPatterns } from './sections/fastapi-production/best-practices-patterns';

// Quiz imports
import { fastapiArchitecturePhilosophyQuiz } from './quizzes/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydanticQuiz } from './quizzes/fastapi-production/request-response-models-pydantic';
import { pathOperationsRoutingQuiz } from './quizzes/fastapi-production/path-operations-routing';
import { dependencyInjectionSystemQuiz } from './quizzes/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemyQuiz } from './quizzes/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2Quiz } from './quizzes/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissionsQuiz } from './quizzes/fastapi-production/authorization-permissions';
import { backgroundTasksQuiz } from './quizzes/fastapi-production/background-tasks';
import { websocketsRealtimeQuiz } from './quizzes/fastapi-production/websockets-realtime';
import { fileUploadsStreamingQuiz } from './quizzes/fastapi-production/file-uploads-streaming';
import { errorHandlingValidationQuiz } from './quizzes/fastapi-production/error-handling-validation';
import { middlewareCorsQuiz } from './quizzes/fastapi-production/middleware-cors';
import { apiDocumentationQuiz } from './quizzes/fastapi-production/api-documentation';
import { testingFastapiQuiz } from './quizzes/fastapi-production/testing-fastapi';
import { asyncPatternsQuiz } from './quizzes/fastapi-production/async-patterns';
import { productionDeploymentQuiz } from './quizzes/fastapi-production/production-deployment';
import { bestPracticesPatternsQuiz } from './quizzes/fastapi-production/best-practices-patterns';

// Multiple choice imports
import { fastapiArchitecturePhilosophyMultipleChoice } from './multiple-choice/fastapi-production/fastapi-architecture-philosophy';
import { requestResponseModelsPydanticMultipleChoice } from './multiple-choice/fastapi-production/request-response-models-pydantic';
import { pathOperationsRoutingMultipleChoice } from './multiple-choice/fastapi-production/path-operations-routing';
import { dependencyInjectionSystemMultipleChoice } from './multiple-choice/fastapi-production/dependency-injection-system';
import { databaseIntegrationSqlalchemyMultipleChoice } from './multiple-choice/fastapi-production/database-integration-sqlalchemy';
import { authenticationJwtOauth2MultipleChoice } from './multiple-choice/fastapi-production/authentication-jwt-oauth2';
import { authorizationPermissionsMultipleChoice } from './multiple-choice/fastapi-production/authorization-permissions';
import { backgroundTasksMultipleChoice } from './multiple-choice/fastapi-production/background-tasks';
import { websocketsRealtimeMultipleChoice } from './multiple-choice/fastapi-production/websockets-realtime';
import { fileUploadsStreamingMultipleChoice } from './multiple-choice/fastapi-production/file-uploads-streaming';
import { errorHandlingValidationMultipleChoice } from './multiple-choice/fastapi-production/error-handling-validation';
import { middlewareCorsMultipleChoice } from './multiple-choice/fastapi-production/middleware-cors';
import { apiDocumentationMultipleChoice } from './multiple-choice/fastapi-production/api-documentation';
import { testingFastapiMultipleChoice } from './multiple-choice/fastapi-production/testing-fastapi';
import { asyncPatternsMultipleChoice } from './multiple-choice/fastapi-production/async-patterns';
import { productionDeploymentMultipleChoice } from './multiple-choice/fastapi-production/production-deployment';
import { bestPracticesPatternsMultipleChoice } from './multiple-choice/fastapi-production/best-practices-patterns';

export const fastapiProductionModule: Module = {
  id: 'fastapi-production',
  title: 'FastAPI Production Mastery',
  description:
    'Master production-ready API development with FastAPI: from architecture and authentication to deployment and best practices. Build scalable, secure, high-performance APIs.',
  icon: '🚀',
  difficulty: 'Advanced',
  estimatedHours: 40,
  topic: 'Python',
  curriculum: 'python',

  sections: [
    {
      id: 'fastapi-architecture-philosophy',
      title: 'FastAPI Architecture & Philosophy',
      content: fastapiArchitecturePhilosophy.content,
      quiz: fastapiArchitecturePhilosophyQuiz,
      multipleChoice: fastapiArchitecturePhilosophyMultipleChoice,
      order: 1,
      estimatedMinutes: 90,
    },
    {
      id: 'request-response-models-pydantic',
      title: 'Request & Response Models (Pydantic)',
      content: requestResponseModelsPydantic.content,
      quiz: requestResponseModelsPydanticQuiz,
      multipleChoice: requestResponseModelsPydanticMultipleChoice,
      order: 2,
      estimatedMinutes: 120,
    },
    {
      id: 'path-operations-routing',
      title: 'Path Operations & Routing',
      content: pathOperationsRouting.content,
      quiz: pathOperationsRoutingQuiz,
      multipleChoice: pathOperationsRoutingMultipleChoice,
      order: 3,
      estimatedMinutes: 100,
    },
    {
      id: 'dependency-injection-system',
      title: 'Dependency Injection System',
      content: dependencyInjectionSystem.content,
      quiz: dependencyInjectionSystemQuiz,
      multipleChoice: dependencyInjectionSystemMultipleChoice,
      order: 4,
      estimatedMinutes: 120,
    },
    {
      id: 'database-integration-sqlalchemy',
      title: 'Database Integration (SQLAlchemy + FastAPI)',
      content: databaseIntegrationSqlalchemy.content,
      quiz: databaseIntegrationSqlalchemyQuiz,
      multipleChoice: databaseIntegrationSqlalchemyMultipleChoice,
      order: 5,
      estimatedMinutes: 150,
    },
    {
      id: 'authentication-jwt-oauth2',
      title: 'Authentication (JWT, OAuth2)',
      content: authenticationJwtOauth2.content,
      quiz: authenticationJwtOauth2Quiz,
      multipleChoice: authenticationJwtOauth2MultipleChoice,
      order: 6,
      estimatedMinutes: 140,
    },
    {
      id: 'authorization-permissions',
      title: 'Authorization & Permissions',
      content: authorizationPermissions.content,
      quiz: authorizationPermissionsQuiz,
      multipleChoice: authorizationPermissionsMultipleChoice,
      order: 7,
      estimatedMinutes: 110,
    },
    {
      id: 'background-tasks',
      title: 'Background Tasks',
      content: backgroundTasks.content,
      quiz: backgroundTasksQuiz,
      multipleChoice: backgroundTasksMultipleChoice,
      order: 8,
      estimatedMinutes: 100,
    },
    {
      id: 'websockets-realtime',
      title: 'WebSockets & Real-Time Communication',
      content: websocketsRealtime.content,
      quiz: websocketsRealtimeQuiz,
      multipleChoice: websocketsRealtimeMultipleChoice,
      order: 9,
      estimatedMinutes: 110,
    },
    {
      id: 'file-uploads-streaming',
      title: 'File Uploads & Streaming Responses',
      content: fileUploadsStreaming.content,
      quiz: fileUploadsStreamingQuiz,
      multipleChoice: fileUploadsStreamingMultipleChoice,
      order: 10,
      estimatedMinutes: 90,
    },
    {
      id: 'error-handling-validation',
      title: 'Error Handling & Validation',
      content: errorHandlingValidation.content,
      quiz: errorHandlingValidationQuiz,
      multipleChoice: errorHandlingValidationMultipleChoice,
      order: 11,
      estimatedMinutes: 100,
    },
    {
      id: 'middleware-cors',
      title: 'Middleware & CORS',
      content: middlewareCors.content,
      quiz: middlewareCorsQuiz,
      multipleChoice: middlewareCorsMultipleChoice,
      order: 12,
      estimatedMinutes: 90,
    },
    {
      id: 'api-documentation',
      title: 'API Documentation (OpenAPI/Swagger)',
      content: apiDocumentation.content,
      quiz: apiDocumentationQuiz,
      multipleChoice: apiDocumentationMultipleChoice,
      order: 13,
      estimatedMinutes: 80,
    },
    {
      id: 'testing-fastapi',
      title: 'Testing FastAPI Applications',
      content: testingFastapi.content,
      quiz: testingFastapiQuiz,
      multipleChoice: testingFastapiMultipleChoice,
      order: 14,
      estimatedMinutes: 120,
    },
    {
      id: 'async-patterns',
      title: 'Async FastAPI Patterns',
      content: asyncPatterns.content,
      quiz: asyncPatternsQuiz,
      multipleChoice: asyncPatternsMultipleChoice,
      order: 15,
      estimatedMinutes: 110,
    },
    {
      id: 'production-deployment',
      title: 'Production Deployment (Uvicorn, Gunicorn)',
      content: productionDeployment.content,
      quiz: productionDeploymentQuiz,
      multipleChoice: productionDeploymentMultipleChoice,
      order: 16,
      estimatedMinutes: 130,
    },
    {
      id: 'best-practices-patterns',
      title: 'FastAPI Best Practices & Patterns',
      content: bestPracticesPatterns.content,
      quiz: bestPracticesPatternsQuiz,
      multipleChoice: bestPracticesPatternsMultipleChoice,
      order: 17,
      estimatedMinutes: 100,
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
      estimatedHours: 10,
    },
    {
      title: 'Real-Time Chat Application',
      description:
        'Implement WebSocket-based chat with FastAPI, including authentication and message persistence',
      difficulty: 'Intermediate',
      estimatedHours: 6,
    },
    {
      title: 'E-Commerce Backend',
      description:
        'Create a production-ready e-commerce API with products, orders, payments, and admin dashboard',
      difficulty: 'Advanced',
      estimatedHours: 15,
    },
    {
      title: 'File Processing Service',
      description:
        'Build an API for uploading, processing, and serving files with background task processing',
      difficulty: 'Intermediate',
      estimatedHours: 5,
    },
  ],

  resources: [
    {
      title: 'Official FastAPI Documentation',
      url: 'https://fastapi.tiangolo.com',
      type: 'documentation',
    },
    {
      title: 'Pydantic Documentation',
      url: 'https://docs.pydantic.dev',
      type: 'documentation',
    },
    {
      title: 'SQLAlchemy Documentation',
      url: 'https://docs.sqlalchemy.org',
      type: 'documentation',
    },
    {
      title: 'Python Type Hints',
      url: 'https://docs.python.org/3/library/typing.html',
      type: 'documentation',
    },
  ],

  tags: [
    'fastapi',
    'python',
    'api',
    'rest',
    'authentication',
    'authorization',
    'sqlalchemy',
    'pydantic',
    'async',
    'production',
    'deployment',
    'testing',
    'security',
    'microservices',
  ],
};
