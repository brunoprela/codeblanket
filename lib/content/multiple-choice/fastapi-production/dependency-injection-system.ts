import { MultipleChoiceQuestion } from '@/lib/types';

export const dependencyInjectionSystemMultipleChoice = [
  {
      id: 'fastapi-di-mc-1',
      question:
        'How does FastAPI handle dependencies that are used multiple times in a single request?',
      options: [
        'Executes the dependency function each time it is used',
        'Caches the result per request and reuses it',
        'Only executes the first dependency and ignores subsequent ones',
        'Raises an error for duplicate dependencies',
      ],
      correctAnswer: 1,
      explanation:
        'FastAPI caches dependency results per request by default. If a dependency is used multiple times (e.g., get_db used by multiple dependencies or endpoints), it\'s executed only once and the result is cached. Example: @app.get("/test") async def test(d1=Depends(get_db), d2=Depends(get_db)) → get_db called once, result reused. This is a major performance optimization. Disable caching with: Depends(get_db, use_cache=False) when you need fresh values (e.g., timestamps, random values).',
    },
    {
      id: 'fastapi-di-mc-2',
      question:
        'What is the correct way to apply a dependency to all routes in an APIRouter?',
      options: [
        'Add Depends() to each route individually',
        'Use dependencies parameter in APIRouter constructor',
        'Use a global middleware',
        'Dependencies cannot be applied to routers',
      ],
      correctAnswer: 1,
      explanation:
        'Use the dependencies parameter in APIRouter constructor to apply dependencies to all routes in that router. Example: router = APIRouter(prefix="/api", dependencies=[Depends(verify_api_key), Depends(rate_limit)]). All routes in this router automatically have these dependencies applied before the route handler runs. This is perfect for: authentication (all routes require auth), rate limiting (all routes limited), logging (all routes logged). No need to add Depends() to each individual route!',
    },
    {
      id: 'fastapi-di-mc-3',
      question: 'What is the purpose of yield in a dependency function?',
      options: [
        'To return multiple values from a dependency',
        'To create a context manager with setup and teardown logic',
        'To make the dependency asynchronous',
        'To cache the dependency result',
      ],
      correctAnswer: 1,
      explanation:
        'yield in dependencies creates a context manager pattern for setup/teardown logic. Code before yield runs before the endpoint (setup), code after yield runs after the endpoint (teardown). Example: def get_db(): db = SessionLocal(); try: yield db; # Endpoint runs here; finally: db.close(); # Always closes even if error. Perfect for: database sessions (open/close), file handles (open/close), locks (acquire/release), connections (connect/disconnect). The finally block ensures cleanup even on exceptions.',
    },
    {
      id: 'fastapi-di-mc-4',
      question: 'How do you override a dependency for testing?',
      options: [
        'Modify the dependency function directly',
        'Use app.dependency_overrides dictionary',
        'Create a new FastAPI app for testing',
        'Dependencies cannot be overridden',
      ],
      correctAnswer: 1,
      explanation:
        'Use app.dependency_overrides dictionary to replace dependencies during testing. Example: def mock_get_db(): return MockDB(); app.dependency_overrides[get_db] = mock_get_db. Now all endpoints using get_db will receive MockDB instead. After tests: app.dependency_overrides.clear(). This is essential for: testing without real database, mocking external APIs, testing auth without real tokens, isolating tests from infrastructure. pytest fixture: @pytest.fixture; def client(): app.dependency_overrides[get_db] = mock_db; yield TestClient(app); app.dependency_overrides.clear().',
    },
    {
      id: 'fastapi-di-mc-5',
      question:
        'What is the benefit of using class-based dependencies over function-based?',
      options: [
        'Class-based dependencies are faster',
        'Classes enable stateful dependencies with methods and properties',
        'Functions cannot be used as dependencies',
        'Classes are required for async dependencies',
      ],
      correctAnswer: 1,
      explanation:
        'Class-based dependencies (callable classes with __init__) provide better organization for complex dependencies with state. Benefits: encapsulate related parameters, provide methods for common operations, use properties for computed values. Example: class Pagination: def __init__(self, page: int = 1, size: int = 20): self.page, self.size = page, size; @property; def offset(self): return (self.page - 1) * self.size; def apply(self, query): return query.offset(self.offset).limit(self.size). Use: @app.get("/") async def root(p: Pagination = Depends()): query = p.apply(query). Better than functions for: complex filtering, pagination, service classes. Performance difference is negligible (<1%).',
    },
  ];
