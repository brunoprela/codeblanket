import { MultipleChoiceQuestion } from '@/lib/types';

export const bestPracticesPatternsMultipleChoice = [
  {
      id: 1,
      question:
        'What is the recommended way to manage configuration and secrets in production FastAPI applications?',
      options: [
        'Use pydantic-settings with environment variables, never hardcode secrets in code, and use secret management services for sensitive values',
        'Store all configuration in config.py file in the repository',
        'Use JSON files for configuration',
        'Hardcode configuration for simplicity',
      ],
      correctAnswer: 0,
      explanation:
        'Production pattern: Use pydantic-settings BaseSettings class that reads from environment variables. Benefits: Type validation, different configs per environment (dev/staging/prod), secrets not in code/repository, can use AWS Secrets Manager/Vault. Pattern: class Settings(BaseSettings): SECRET_KEY: str; DATABASE_URL: str; class Config: env_file = ".env". Never commit .env to git (add to .gitignore). For production: Set env vars in Kubernetes secrets or cloud provider. Storing in code (options 2, 4) exposes secrets in version control. JSON files (option 3) still need to be secured and excluded from git.',
    },
    {
      id: 2,
      question: 'Why should you use response_model in FastAPI endpoints?',
      options: [
        'response_model validates and serializes output, ensures consistent API responses, and prevents accidentally leaking sensitive fields like passwords',
        'response_model is only for documentation',
        'response_model makes responses faster',
        'response_model is optional and not important',
      ],
      correctAnswer: 0,
      explanation:
        'response_model provides output validation and filtering. Example: @app.get("/users/{id}", response_model=UserPublic) - Even if User model has password_hash field, UserPublic (without password_hash) ensures it\'s never returned. Benefits: 1) Security: Filter sensitive fields, 2) Consistency: All endpoints return predictable structure, 3) Documentation: OpenAPI schema accurate, 4) Validation: Ensures response matches spec. Not just documentation (option 2) - actively filters/validates. Doesn\'t affect speed (option 3). Critical for security (option 4) - prevents leaking database IDs, internal fields, passwords.',
    },
    {
      id: 3,
      question:
        'What is the purpose of using connection pooling for database connections?',
      options: [
        'Connection pooling reuses database connections instead of creating new ones for each request, dramatically reducing connection overhead and improving performance',
        'Connection pooling increases the number of available connections',
        'Connection pooling is only for PostgreSQL',
        'Connection pooling stores query results',
      ],
      correctAnswer: 0,
      explanation:
        "Creating database connection is expensive (TCP handshake, authentication, ~50-100ms). Without pooling: Every request creates new connection, 1000 requests = 1000 connections created/destroyed, massive overhead. With pooling: Pre-create 20 connections (pool), reuse connections across requests, request borrows connection from pool, returns after use. Result: 1000 requests = 20 connections reused, 50x faster. Configuration: create_engine(pool_size=20, max_overflow=0). Doesn't increase connections (option 2) - manages reuse. Works with all databases (option 3). Doesn't cache results (option 4) - that's application caching (Redis).",
    },
    {
      id: 4,
      question: 'What is an API anti-pattern you should avoid in FastAPI?',
      options: [
        'Exposing internal errors to clients, which leaks system details and helps attackers understand your infrastructure',
        'Using async endpoints',
        'Validating user input',
        'Implementing error handling',
      ],
      correctAnswer: 0,
      explanation:
        'Anti-pattern: except Exception as e: return {"error": str(e)}. Problem: Exposes stack traces, file paths (/app/src/auth/handlers.py), database details (PostgreSQL version), library versions (SQLAlchemy 2.0). Attackers use this to: Identify vulnerabilities, craft targeted attacks, understand architecture. Production pattern: Log full error server-side, return generic message to client: return {"error": "An error occurred", "request_id": "abc123"}. User references request_id in support tickets, engineers find full details in logs. Using async (option 2), validation (option 3), and error handling (option 4) are all best practices, not anti-patterns.',
    },
    {
      id: 5,
      question:
        'Why should you use code quality tools like Black, isort, ruff, and mypy in FastAPI projects?',
      options: [
        'These tools enforce consistent code style, catch bugs early with type checking, and improve code maintainability across the team',
        'They make the code run faster',
        'They are required for FastAPI to work',
        'They automatically fix all bugs',
      ],
      correctAnswer: 0,
      explanation:
        "Code quality tool benefits: Black (consistent formatting, no style debates), isort (organized imports), ruff (fast linting, catches bugs like unused variables), mypy (type checking catches type errors before runtime). Benefits: Team consistency (code looks uniform), catch bugs early (mypy finds type mismatches), faster reviews (no style discussions), easier onboarding (predictable code). Setup: pre-commit hooks run tools automatically on git commit. They don't affect runtime performance (option 2), aren't required by FastAPI (option 3), and don't fix logic bugs automatically (option 4) - they catch potential issues and enforce standards. Production pattern: Run in CI/CD, block merge if checks fail.",
    },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
