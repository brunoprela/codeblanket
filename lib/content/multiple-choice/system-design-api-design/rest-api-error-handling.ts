/**
 * Multiple choice questions for REST API Error Handling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const restapierrorhandlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'error-q1',
    question:
      "A user tries to delete a post they don't own. Which status code and error are most appropriate?",
    options: [
      '401 Unauthorized with "Not authenticated"',
      '403 Forbidden with "You don\'t have permission to delete this post"',
      '404 Not Found with "Post not found"',
      '400 Bad Request with "Invalid post ID"',
    ],
    correctAnswer: 1,
    explanation:
      "403 Forbidden is for authenticated users who lack permission for a specific resource. 401 is for authentication issues, 404 would leak information (post exists but you can't access it), 400 is for malformed requests.",
  },
  {
    id: 'error-q2',
    question:
      'Your API encounters a database connection error. What should you return to the client?',
    options: [
      '500 Internal Server Error with database connection details',
      '500 Internal Server Error with generic message and request ID',
      '503 Service Unavailable with database error message',
      '400 Bad Request with "Database error"',
    ],
    correctAnswer: 1,
    explanation:
      '500 with generic message protects internal details. Include request ID for support to investigate. Never expose database details (security risk). 503 is for planned maintenance/temporary unavailability. 400 is for client errors.',
  },
  {
    id: 'error-q3',
    question:
      'A client exceeds rate limit (1000 req/hour). Which headers should your 429 response include?',
    options: [
      'Only X-RateLimit-Limit header',
      'X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After',
      'Only Retry-After header',
      'No special headers needed with 429',
    ],
    correctAnswer: 1,
    explanation:
      'Provide complete rate limit information: total limit, remaining requests (0), when limit resets (timestamp), and retry-after (seconds). This helps clients implement proper backoff strategies.',
  },
  {
    id: 'error-q4',
    question:
      'User submits registration form with invalid email and short password. Best error response structure?',
    options: [
      'Single error message: "Invalid email or password"',
      'Array of errors with field-level details for each validation failure',
      'Two separate 400 responses (one per field)',
      'Generic "Validation failed" message',
    ],
    correctAnswer: 1,
    explanation:
      "Return field-level details in single response so client can show all errors at once. Users shouldn't fix one field, resubmit, then discover another error. Good UX requires comprehensive validation feedback.",
  },
  {
    id: 'error-q5',
    question: 'What is the main purpose of idempotency keys in error handling?',
    options: [
      'To encrypt error messages',
      'To allow safe retries of non-idempotent operations',
      'To track error frequency',
      'To validate request authenticity',
    ],
    correctAnswer: 1,
    explanation:
      'Idempotency keys let clients safely retry operations like payments without duplicates. Server caches result by key and returns cached response for retries. This solves network timeout uncertainty ("did my payment go through?").',
  },
];
