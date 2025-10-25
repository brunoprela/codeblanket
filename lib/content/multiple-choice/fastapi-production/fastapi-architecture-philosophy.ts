import { MultipleChoiceQuestion } from '@/lib/types';

export const fastapiArchitecturePhilosophyMultipleChoice = [
  {
      id: 'fastapi-arch-mc-1',
      question:
        'What is the primary reason FastAPI achieves performance comparable to NodeJS and Go?',
      options: [
        'It is written in C++',
        'It uses ASGI with async/await for non-blocking I/O operations',
        'It has a smaller codebase than other frameworks',
        'It caches all responses automatically',
      ],
      correctAnswer: 1,
      explanation:
        "FastAPI achieves high performance through ASGI (Asynchronous Server Gateway Interface) which enables async/await patterns for non-blocking I/O operations. This allows it to handle thousands of concurrent connections efficiently, similar to NodeJS event loop and Go goroutines. FastAPI + Uvicorn can handle ~37,000 req/s vs Flask ~3,000 req/s. The async architecture is key—requests don't block threads while waiting for I/O (database, HTTP calls).",
    },
    {
      id: 'fastapi-arch-mc-2',
      question:
        'What does FastAPI automatically generate from Python type hints?',
      options: [
        'Database migrations',
        'OpenAPI documentation, request validation, and response serialization',
        'Test cases',
        'Frontend code',
      ],
      correctAnswer: 1,
      explanation:
        "FastAPI leverages Python type hints to automatically generate: (1) OpenAPI documentation (Swagger UI at /docs), (2) Request validation (Pydantic validates incoming data), (3) Response serialization (automatic JSON conversion), (4) JSON Schema for API contracts. This is FastAPI's superpower—types are the single source of truth for validation, docs, and IDE support. You don't write docs separately; they're generated from your code.",
    },
    {
      id: 'fastapi-arch-mc-3',
      question:
        "Which component provides FastAPI's data validation capabilities?",
      options: ['Starlette', 'Uvicorn', 'Pydantic', 'SQLAlchemy'],
      correctAnswer: 2,
      explanation:
        "Pydantic provides FastAPI's data validation. Pydantic uses Python type hints to validate, serialize, and deserialize data. In FastAPI: request bodies are validated by Pydantic models, responses are serialized by Pydantic, Field validators enforce constraints. Pydantic v2 uses Rust-based validation (pydantic-core) for 5-17x better performance. Starlette provides ASGI routing, Uvicorn is the ASGI server, SQLAlchemy is the ORM.",
    },
    {
      id: 'fastapi-arch-mc-4',
      question:
        'When comparing FastAPI, Flask, and Django REST Framework, which statement is most accurate?',
      options: [
        'FastAPI is always the best choice for any project',
        'Flask is faster than FastAPI for I/O-bound operations',
        'FastAPI excels at APIs/microservices, Django at full web apps, Flask at simple/legacy APIs',
        'Django REST Framework has better performance than FastAPI',
      ],
      correctAnswer: 2,
      explanation:
        'Each framework has optimal use cases: FastAPI excels at: APIs, microservices, high-performance needs, async architecture, automatic documentation. Django/DRF excels at: Full web applications (HTML + API), admin panel needed, batteries-included features. Flask excels at: Simple synchronous APIs, legacy systems, flexibility with extensions. Performance: FastAPI >> Flask >> Django for I/O-bound ops. The "best" framework depends on project requirements: new API → FastAPI, full web app → Django, simple API → Flask.',
    },
    {
      id: 'fastapi-arch-mc-5',
      question: 'What is the purpose of dependency injection in FastAPI?',
      options: [
        'To automatically install Python packages',
        'To manage and inject reusable dependencies like database sessions, authentication, and shared logic',
        'To inject CSS into HTML templates',
        'To automatically create API documentation',
      ],
      correctAnswer: 1,
      explanation:
        'Dependency injection in FastAPI manages reusable dependencies using the Depends() function. Common uses: (1) Database sessions: db: Session = Depends(get_db), (2) Authentication: user: User = Depends(get_current_user), (3) Shared logic: pagination, rate limiting, permissions. Benefits: Reusable code, testable (inject mocks), automatic resolution, type-safe. Dependencies can depend on other dependencies (dependency trees). This is FastAPI\'s "secret weapon" for clean, maintainable code.',
    },
  ];
