import { MultipleChoiceQuestion } from '@/lib/types';

export const pathOperationsRoutingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fastapi-routing-mc-1',
    question:
      'What is the correct HTTP method for partially updating a resource?',
    options: [
      'POST - because it modifies data',
      'PUT - because it updates the resource',
      'PATCH - for partial updates',
      'UPDATE - for modifying existing resources',
    ],
    correctAnswer: 2,
    explanation:
      'PATCH is the correct HTTP method for partial updates. PATCH updates only the provided fields, leaving others unchanged. Example: PATCH /users/123 with {"email": "new@example.com"} updates only email. PUT replaces the entire resource—all fields must be provided. POST creates new resources. There is no UPDATE method in HTTP. RESTful API design: POST (create), GET (read), PUT (replace), PATCH (update), DELETE (delete).',
  },
  {
    id: 'fastapi-routing-mc-2',
    question:
      'Why must static routes be defined before dynamic routes in FastAPI?',
    options: [
      'For better performance',
      'Static routes have priority in path matching—FastAPI matches in definition order',
      'It is a FastAPI limitation',
      "There is no requirement—order doesn't matter",
    ],
    correctAnswer: 1,
    explanation:
      'FastAPI matches routes in the order they are defined. If you define @app.get("/users/{user_id}") before @app.get("/users/me"), then GET /users/me will match the dynamic route (user_id="me") instead of the static route. Correct order: 1. Static routes first: @app.get("/users/me"), 2. Dynamic routes second: @app.get("/users/{user_id}"). This ensures /users/me hits the specific handler, while /users/123 hits the dynamic handler. Always define specific/static routes before generic/dynamic ones.',
  },
  {
    id: 'fastapi-routing-mc-3',
    question: 'What is the purpose of APIRouter in FastAPI?',
    options: [
      'To handle HTTP routing in the web server',
      'To organize and group related endpoints with shared configuration (prefix, tags, dependencies)',
      'To automatically generate API documentation',
      'To validate request parameters',
    ],
    correctAnswer: 1,
    explanation:
      'APIRouter groups related endpoints with shared configuration: prefix (e.g., "/users"), tags (for OpenAPI docs grouping), dependencies (auth, rate limiting applied to all routes), responses (common error responses). Example: users_router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(verify_auth)]). Benefits: organize large APIs into modules, apply shared logic (auth, logging) at router level, separate concerns (users router, products router), include in main app: app.include_router(users_router). Routes become: GET /users/, GET /users/{id}, etc.',
  },
  {
    id: 'fastapi-routing-mc-4',
    question: 'What does the status_code parameter do in a route decorator?',
    options: [
      'Validates incoming request status',
      'Sets the HTTP status code for successful responses',
      'Catches errors with specific status codes',
      'Defines allowed status codes for the endpoint',
    ],
    correctAnswer: 1,
    explanation:
      'status_code parameter sets the HTTP status code returned for successful responses. Default is 200 OK. Common patterns: @app.post("/users", status_code=201) → Returns 201 Created for POST requests (new resource), @app.delete("/users/{id}", status_code=204) → Returns 204 No Content for DELETE (no body), @app.get("/users", status_code=200) → Returns 200 OK (default, can omit). Use appropriate status codes: 200 (OK - GET, PUT, PATCH), 201 (Created - POST), 204 (No Content - DELETE), 4xx (client errors), 5xx (server errors). This makes APIs RESTful and predictable.',
  },
  {
    id: 'fastapi-routing-mc-5',
    question:
      'How do you capture a file path (with slashes) as a path parameter?',
    options: [
      'Use regular string parameter: {path}',
      'Use special syntax: {path:path}',
      'URL encode the slashes first',
      'FastAPI does not support this',
    ],
    correctAnswer: 1,
    explanation:
      '{path:path} syntax allows capturing paths with slashes. Example: @app.get("/files/{file_path:path}") allows GET /files/docs/api/readme.md where file_path = "docs/api/readme.md". Without :path, slashes would be interpreted as separate path segments and cause 404. Regular parameter {file_path} would only match until first slash. Use cases: file serving, nested resource paths, proxy endpoints. Alternative: URL encode slashes (%2F), but :path is cleaner and more idiomatic.',
  },
];
