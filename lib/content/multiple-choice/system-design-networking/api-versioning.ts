/**
 * Multiple choice questions for API Versioning section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apiversioningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'api-version-strategy',
    question:
      'Which API versioning strategy is most commonly used for public REST APIs?',
    options: [
      'Query parameter versioning (?version=2)',
      'URL path versioning (/api/v2/users)',
      'Header versioning (API-Version: 2)',
      'Content negotiation (Accept: application/vnd.api.v2+json)',
    ],
    correctAnswer: 1,
    explanation:
      "URL path versioning (/api/v2/users) is most common for public REST APIs because it's explicit, easy to test with curl/browsers, cacheable, and doesn't require special headers. Query parameters pollute URLs, headers are less discoverable, and content negotiation is complex for clients.",
  },
  {
    id: 'api-breaking-change',
    question:
      'Which change to an API is considered a breaking change requiring a new major version?',
    options: [
      'Adding a new optional field to the response',
      'Adding a new endpoint',
      'Renaming an existing response field from "name" to "fullName"',
      'Adding a new optional query parameter',
    ],
    correctAnswer: 2,
    explanation:
      "Renaming a field is breaking because existing clients expect the old field name and will break when it's removed. Adding new optional fields, endpoints, or parameters are additive changes that are backward compatible (old clients can ignore them).",
  },
  {
    id: 'api-deprecation-header',
    question:
      'Which HTTP header should an API return to indicate that an endpoint is deprecated and will be removed on a specific date?',
    options: [
      'X-API-Deprecated: true',
      'Warning: 299 - "Deprecated API"',
      'Sunset: Sat, 31 Dec 2024 23:59:59 GMT',
      'Cache-Control: no-cache',
    ],
    correctAnswer: 2,
    explanation:
      'The Sunset header (RFC 8594) indicates when a resource will be removed, using an HTTP date format. Deprecation header indicates current deprecation status, but Sunset provides the critical removal date. Warning is for cache warnings, and Cache-Control is for caching directives.',
  },
  {
    id: 'api-semantic-versioning',
    question:
      'In semantic versioning (MAJOR.MINOR.PATCH), when should you increment the MINOR version?',
    options: [
      'When making backward-incompatible changes',
      'When adding new features in a backward-compatible manner',
      'When fixing bugs without changing functionality',
      'When updating documentation',
    ],
    correctAnswer: 1,
    explanation:
      "MINOR version increments for new backward-compatible features. MAJOR increments for breaking changes. PATCH increments for backward-compatible bug fixes. Documentation updates don't require version changes.",
  },
  {
    id: 'api-graphql-versioning',
    question: 'How does GraphQL handle API versioning differently from REST?',
    options: [
      'GraphQL uses URL path versioning like /graphql/v2',
      'GraphQL uses header versioning with GraphQL-Version header',
      'GraphQL avoids versioning by deprecating fields and evolving the schema gradually',
      'GraphQL requires a new schema file for each version',
    ],
    correctAnswer: 2,
    explanation:
      "GraphQL philosophy is to avoid versioning by evolving the schema gradually. Fields are marked with @deprecated directive rather than removed, allowing old and new clients to coexist. Clients request only the fields they need, so new fields don't break old clients.",
  },
];
