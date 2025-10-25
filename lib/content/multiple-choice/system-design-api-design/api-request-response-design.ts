/**
 * Multiple choice questions for API Request/Response Design section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apirequestresponsedesignMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'req-res-q1',
      question:
        'Which pagination approach best handles real-time data where items are frequently added/removed?',
      options: [
        'Offset-based: GET /posts?page=2&limit=20',
        'Cursor-based: GET /posts?cursor=abc123&limit=20',
        'Page numbers with caching',
        'Load all data client-side',
      ],
      correctAnswer: 1,
      explanation:
        'Cursor-based pagination uses stable references (like ID + timestamp), preventing duplicates or missing items when data changes. Offset-based can show duplicates if items are inserted between requests.',
    },
    {
      id: 'req-res-q2',
      question:
        "What\'s the correct HTTP status code and response for an invalid email during registration?",
      options: [
        '200 OK with {"success": false}',
        '400 Bad Request with structured error',
        '500 Internal Server Error',
        '422 Unprocessable Entity',
      ],
      correctAnswer: 1,
      explanation:
        '400 Bad Request is appropriate for client input errors. While 422 is also valid for validation errors, 400 is more commonly used and understood.',
    },
    {
      id: 'req-res-q3',
      question:
        'Which query syntax best supports filtering users by age range (18-65)?',
      options: [
        'GET /users?age=18-65',
        'GET /users?minAge=18&maxAge=65',
        'GET /users?age[gte]=18&age[lte]=65',
        'GET /users?filter=age BETWEEN 18 AND 65',
      ],
      correctAnswer: 2,
      explanation:
        'Operator-based filtering (age[gte]=18) is extensible, consistent, and widely used (MongoDB-style). It scales to other operators (gt, lt, ne, in) without parameter explosion.',
    },
    {
      id: 'req-res-q4',
      question:
        'A mobile app needs only id, name, avatar but not full profile. Best approach?',
      options: [
        'Create /users/123/mobile endpoint',
        'Use GET /users/123?fields=id,name,avatar',
        'Return everything, filter client-side',
        'Require migration to GraphQL',
      ],
      correctAnswer: 1,
      explanation:
        'Field selection (sparse fieldsets) is the REST way to optimize bandwidth without creating multiple endpoints or paradigm shifts.',
    },
    {
      id: 'req-res-q5',
      question: 'Why is OFFSET 10000 slow in pagination queries?',
      options: [
        'Database needs more memory',
        'Database must scan and skip first 10,000 rows',
        'Query optimizer is not configured',
        'Index is missing on the ID column',
      ],
      correctAnswer: 1,
      explanation:
        'OFFSET requires scanning and discarding rows. Cursor-based pagination uses WHERE id > X which uses indexes efficiently regardless of position.',
    },
  ];
