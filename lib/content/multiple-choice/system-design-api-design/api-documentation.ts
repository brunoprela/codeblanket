/**
 * Multiple choice questions for API Documentation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apidocumentationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'docs-q1',
    question: 'What is OpenAPI (Swagger) and why is it important?',
    options: [
      'A programming language for writing APIs',
      'A standard specification format for describing RESTful APIs',
      'A database for storing API documentation',
      'A testing framework for APIs',
    ],
    correctAnswer: 1,
    explanation:
      "OpenAPI (formerly Swagger) is a standard specification format for describing REST APIs. It's language-agnostic, machine-readable, and enables auto-generation of documentation, client libraries, and test cases. Industry standard for API documentation.",
    difficulty: 'easy',
  },
  {
    id: 'docs-q2',
    question:
      'Why provide code examples in multiple languages in API documentation?',
    options: [
      'To make the documentation look longer',
      'To help developers in different ecosystems quickly integrate',
      'To test the API in all languages',
      "It's required by OpenAPI specification",
    ],
    correctAnswer: 1,
    explanation:
      'Code examples in multiple languages (JavaScript, Python, cURL, etc.) reduce integration time significantly. Developers can copy-paste working examples instead of translating from documentation. Not required by spec, but greatly improves developer experience.',
    difficulty: 'easy',
  },
  {
    id: 'docs-q3',
    question:
      'What is the benefit of interactive API documentation (Swagger UI, Redoc)?',
    options: [
      'It makes documentation load faster',
      'Developers can test API calls directly from the documentation',
      'It automatically fixes API bugs',
      'It reduces server costs',
    ],
    correctAnswer: 1,
    explanation:
      'Interactive documentation allows developers to make API calls directly from the docs without writing code first. They can test authentication, see live responses, and understand the API faster. Swagger UI and Redoc provide this interactivity.',
    difficulty: 'easy',
  },
  {
    id: 'docs-q4',
    question:
      'Why should API documentation be auto-generated from code when possible?',
    options: [
      'Auto-generated docs are always better than manual',
      'It ensures documentation stays in sync with actual API behavior',
      'It reduces server hosting costs',
      'It makes the API faster',
    ],
    correctAnswer: 1,
    explanation:
      'Auto-generating documentation from code (e.g., using annotations/decorators) ensures docs stay up-to-date when API changes. Manual docs often become outdated, causing developer frustration. Tools like Swagger, FastAPI, and NestJS provide auto-generation.',
    difficulty: 'medium',
  },
  {
    id: 'docs-q5',
    question:
      'What should comprehensive API documentation include beyond endpoint descriptions?',
    options: [
      'Only request/response schemas',
      'Authentication, rate limits, error codes, pagination, code examples',
      'Just the OpenAPI spec file',
      'Only success responses',
    ],
    correctAnswer: 1,
    explanation:
      'Complete API docs need: authentication methods, rate limits, all error codes, pagination patterns, code examples in multiple languages, and edge cases. Just schemas are insufficient. Document the full developer experience.',
    difficulty: 'easy',
  },
];
