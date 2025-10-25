import { MultipleChoiceQuestion } from '@/lib/types';

export const errorHandlingValidationMultipleChoice = [
  {
    id: 1,
    question:
      "What is the correct HTTP status code for a request where the user is authenticated but doesn't have permission to access the resource?",
    options: [
      '403 Forbidden - The user is authenticated but not authorized to access this resource',
      '401 Unauthorized - The user needs to authenticate',
      '404 Not Found - Hide the existence of the resource',
      '400 Bad Request - The request is invalid',
    ],
    correctAnswer: 0,
    explanation:
      '403 Forbidden indicates the server understood the request and the user is authenticated, but refuses to authorize it due to insufficient permissions. Common scenario: Regular user trying to access admin endpoint. 401 Unauthorized (option 2) means authentication is required but not provided (no token, invalid token). 404 Not Found (option 3) is sometimes used to hide resource existence for security, but this violates HTTP semantics—use 403 and return "Access denied" message. 400 Bad Request (option 4) is for malformed requests, not authorization. The distinction: 401 = "who are you?" (authentication), 403 = "I know who you are, but you can\'t do this" (authorization). Example: GET /admin/users with valid user token → 403 (authenticated but not admin). No token → 401 (not authenticated).',
  },
  {
    id: 2,
    question:
      'Why should production error responses avoid exposing internal details like stack traces or database errors?',
    options: [
      'Internal details provide attackers with information about system architecture, technologies used, and potential vulnerabilities to exploit',
      'Stack traces make the API responses too large and slow',
      'Internal details are harder for frontend developers to parse',
      "Error messages with stack traces don't work with JSON format",
    ],
    correctAnswer: 0,
    explanation:
      'Exposing internal details is a critical security vulnerability. Stack traces reveal: 1) Programming language and version (Python 3.11), 2) Framework and libraries (FastAPI 0.104.1), 3) File paths and directory structure (/app/src/auth/handlers.py), 4) Database technology (PostgreSQL connection error), 5) Third-party services (Redis, S3 bucket names). Attackers use this information to: identify known vulnerabilities in specific versions, craft targeted attacks, understand system architecture. Example: Database error revealing "PostgreSQL 9.6" → attacker knows to try SQL injection techniques specific to that version. Production practice: Log full details server-side (for debugging), return generic message to client ("An error occurred", request_id for correlation). Options 2, 3, 4 are not the primary concerns—security is. Development: show details for debugging. Production: hide everything, provide request_id for support to correlate with server logs.',
  },
  {
    id: 3,
    question:
      'What is the purpose of including a request ID in error responses?',
    options: [
      'To correlate frontend error reports with backend logs, enabling developers to find full details including stack traces and context',
      'To track how many errors each user encounters',
      'To automatically retry failed requests',
      'To encrypt error messages for security',
    ],
    correctAnswer: 0,
    explanation:
      'Request IDs enable error correlation between frontend and backend. Scenario: User reports "I got an error creating an order." Without request_id, you search through millions of log lines. With request_id: 1) Frontend shows error with request_id: "Error a1b2c3d4", 2) User reports: "Error a1b2c3d4", 3) Developer searches logs: grep "a1b2c3d4" logs.txt, 4) Finds full context: stack trace, user_id, request params, database query. Implementation: Middleware generates UUID for each request, adds to response headers and error messages, logs include request_id. Benefits: Fast debugging (find needle in haystack), user privacy (don\'t ask user for sensitive request details), support efficiency (reference error by ID in tickets). Request IDs don\'t track error counts per user (option 2—use analytics), don\'t retry requests (option 3—separate retry logic), don\'t encrypt (option 4—use TLS). Pattern: {"error": "...", "request_id": "a1b2c3d4", "message": "..."}, user tells support "request_id: a1b2c3d4".',
  },
  {
    id: 4,
    question:
      'When should you use HTTP status code 422 (Unprocessable Entity) instead of 400 (Bad Request)?',
    options: [
      '422 is for requests that are syntactically correct but fail semantic validation (e.g., email format invalid), while 400 is for malformed requests (e.g., invalid JSON)',
      '422 is faster to process than 400',
      '422 is only used for file uploads',
      '422 and 400 are interchangeable and can be used for any client error',
    ],
    correctAnswer: 0,
    explanation:
      '422 Unprocessable Entity is specifically for semantic validation failures, while 400 Bad Request is for syntactic errors. Examples: 400 Bad Request: Invalid JSON syntax ({ missing closing brace), Missing required Content-Type header, Malformed URL encoding. 422 Unprocessable Entity: Email field is "not-an-email" (valid string, invalid email format), Age is -5 (valid integer, invalid value for age), Password is "123" (too short, fails validation rules). The distinction: 400 = "I can\'t parse your request", 422 = "I understand your request but the data doesn\'t meet requirements". FastAPI automatically returns 422 for Pydantic validation failures. When to use each: 400 for request structure problems, 422 for data validation problems. This helps clients distinguish: 400 → fix request format (programming error), 422 → fix input data (user error). Options 2, 3, 4 are incorrect—the distinction is specifically about validation type, not performance, file uploads, or interchangeability.',
  },
  {
    id: 5,
    question:
      'What is the best practice for handling Pydantic validation errors in FastAPI error responses?',
    options: [
      'Create a custom exception handler that formats validation errors into a structured response with field names, error messages, and error types',
      'Let FastAPI return the default validation error response',
      'Return a generic 400 error without details',
      'Validation errors should return 500 status code',
    ],
    correctAnswer: 0,
    explanation:
      'Custom validation error handlers provide better UX by formatting errors clearly. Default FastAPI validation response: {"detail": [{"loc": ["body", "email"], "msg": "field required", "type": "value_error.missing"}]}. Not user-friendly! Custom formatted response: {"error": "validation_error", "message": "Invalid input", "details": [{"field": "email", "message": "Email is required", "code": "value_error.missing"}]}. Benefits: 1) Clean field names (email vs body.email), 2) Human-readable messages, 3) Frontend can map errors to form fields, 4) Error codes for programmatic handling. Implementation: @app.exception_handler(RequestValidationError) to intercept and reformat. Letting FastAPI use default (option 2) works but UX suffers. Returning generic 400 without details (option 3) prevents users from fixing issues. Using 500 (option 4) is wrong—validation is client error (4xx), not server error (5xx). Production pattern: Custom handler formats errors, includes request_id, maintains error context for frontend form field highlighting.',
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
