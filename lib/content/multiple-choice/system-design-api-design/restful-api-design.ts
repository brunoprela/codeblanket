/**
 * Multiple choice questions for RESTful API Design Principles section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const restfulapidesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'rest-q1',
    question:
      "Which HTTP method should be used to partially update a user's email address without affecting other fields?",
    options: [
      'POST /api/users/123 with {"email": "new@example.com"}',
      'PUT /api/users/123 with {"email": "new@example.com"}',
      'PATCH /api/users/123 with {"email": "new@example.com"}',
      'GET /api/users/123/update-email?email=new@example.com',
    ],
    correctAnswer: 2,
    explanation:
      'PATCH is designed for partial updates, modifying only the specified fields. PUT would replace the entire resource, potentially clearing other fields. POST is for creating resources, and GET should never modify data.',
  },
  {
    id: 'rest-q2',
    question:
      'A client retries a failed DELETE request 3 times due to network issues. The resource was actually deleted on the first attempt. What makes this safe?',
    options: [
      'DELETE operations always return 404 after the first success',
      'DELETE is idempotent - multiple deletions have the same effect',
      'The server stores a request ID to prevent duplicates',
      'DELETE operations are automatically cached by HTTP',
    ],
    correctAnswer: 1,
    explanation:
      'DELETE is idempotent by design. Whether you delete a resource once or multiple times, the end state is the same - the resource is gone. This makes retries safe. The second attempt typically returns 404 (not found) or 204 (no content), but the key is that the operation is idempotent.',
  },
  {
    id: 'rest-q3',
    question:
      'Which URL follows RESTful naming conventions for retrieving the 5 most recent orders of a specific customer?',
    options: [
      'GET /api/getCustomerOrders?customerId=123&limit=5',
      'GET /api/customer/123/recent-orders',
      'GET /api/customers/123/orders?sort=created_at:desc&limit=5',
      'POST /api/customers/orders with {"customerId": 123, "limit": 5}',
    ],
    correctAnswer: 2,
    explanation:
      'Option 3 follows REST conventions: uses plural nouns (customers, orders), hierarchical structure showing relationship, GET method for retrieval, and query parameters for filtering/sorting. Option 1 uses verbs in URL, option 2 uses singular and vague naming, option 4 incorrectly uses POST for reading data.',
  },
  {
    id: 'rest-q4',
    question:
      'What is the main principle of the "stateless" constraint in REST?',
    options: [
      'The API should not store any data in a database',
      'Each request must contain all information needed to process it',
      'The server should not maintain any state at all',
      'Clients cannot maintain session cookies or tokens',
    ],
    correctAnswer: 1,
    explanation:
      'Stateless means each request contains all information needed (auth tokens, parameters, etc.) - the server does not store session state between requests. The server can still have databases (option 1 wrong), maintain application state like data (option 3 wrong), and clients can hold tokens (option 4 wrong). The key is no server-side session state.',
  },
  {
    id: 'rest-q5',
    question:
      'According to the Richardson Maturity Model, what distinguishes a Level 2 REST API from Level 3?',
    options: [
      'Level 2 uses JSON while Level 3 uses XML',
      'Level 2 uses HTTP verbs correctly, Level 3 adds hypermedia controls (HATEOAS)',
      'Level 2 supports pagination while Level 3 supports filtering',
      'Level 2 is synchronous while Level 3 supports async operations',
    ],
    correctAnswer: 1,
    explanation:
      'The Richardson Maturity Model levels are: Level 0 (single endpoint), Level 1 (multiple resources), Level 2 (HTTP verbs and status codes correctly), Level 3 (adds HATEOAS - responses include links to related resources). Most production APIs are Level 2. The other options are unrelated to the maturity model.',
  },
];
