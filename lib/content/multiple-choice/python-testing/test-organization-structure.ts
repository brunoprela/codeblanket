import { MultipleChoiceQuestion } from '@/lib/types';

export const testOrganizationStructureMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tos-mc-1',
      question:
        'What is the recommended test directory structure pattern for most production applications?',
      options: [
        'Flat structure: All tests in tests/ directory',
        'Mirrored structure: tests/ mirrors src/ directory organization',
        'Test type separation: tests/unit/, tests/integration/, tests/e2e/',
        'Feature-based: tests organized by product features, not code structure',
      ],
      correctAnswer: 1,
      explanation:
        'Mirrored structure (tests/ mirrors src/) is recommended for most production applications: Intuitive: If src/api/routes.py exists, test is at tests/api/test_routes.py. Easy navigation: Know exactly where to find or add tests. Scales well: Grows naturally as codebase grows. Maintains context: Related tests grouped together. Flat structure only works for small projects (<1,000 tests). Test type separation (unit/integration/e2e) is better for large projects (>10K tests) but adds complexity. Feature-based organization is less common and harder to maintain when code structure changes.',
    },
    {
      id: 'tos-mc-2',
      question: 'What is the primary purpose of conftest.py in pytest?',
      options: [
        'To configure pytest command-line options and test discovery',
        'To define shared fixtures available to all tests without imports',
        'To list all test files that should be executed by pytest',
        'To store environment variables and secrets for testing',
      ],
      correctAnswer: 1,
      explanation:
        'conftest.py defines shared fixtures available to all tests in its directory and subdirectories—no imports needed: pytest automatically discovers conftest.py files. Fixtures defined here are automatically available to all test files. Example: @pytest.fixture in conftest.py → use in any test without import. Scope: Root conftest.py affects all tests, tests/unit/conftest.py affects only unit tests. Benefits: Eliminate fixture duplication (define once, use everywhere), maintain consistency, centralize test setup. pytest.ini handles configuration/options. Environment variables typically in .env or environment, not conftest.py.',
    },
    {
      id: 'tos-mc-3',
      question: 'When organizing tests with classes, what is the main benefit?',
      options: [
        'Classes make tests run faster than function-based tests',
        'Classes allow logical grouping of related tests with shared setup',
        'Classes are required by pytest for test discovery',
        'Classes reduce the total number of assertions needed',
      ],
      correctAnswer: 1,
      explanation:
        "Classes provide logical grouping of related tests with shared setup: Group related tests: TestPaymentCreation, TestPaymentProcessing, TestPaymentRefunds. Shared setup: Class-level fixtures for common initialization. Clear hierarchy: Test reports show class structure (better organization). Navigation: Jump to specific test group easily. No performance benefit: Classes don't run faster than functions. Not required: pytest works with both functions and classes. Assertions: Number of assertions is independent of test organization. Use classes for >5 related tests with shared setup; use functions for simple, independent tests.",
    },
    {
      id: 'tos-mc-4',
      question:
        'For a large test suite (5,000+ tests), which organization pattern enables the fastest CI/CD feedback?',
      options: [
        'Flat structure with all tests in one directory',
        'Test type separation: tests/unit/, tests/integration/, tests/e2e/',
        'Alphabetical organization: tests/a-f/, tests/g-m/, tests/n-z/',
        'Random distribution across multiple test directories',
      ],
      correctAnswer: 1,
      explanation:
        "Test type separation enables selective execution for fast CI/CD: Unit tests (60-75%): Fast (<5 min), run on every commit. Integration tests (20-30%): Medium speed (10 min), run before merge. E2E tests (5-10%): Slow (30+ min), run nightly or on main branch. Selective execution: pytest tests/unit (fast feedback), pytest tests/integration (comprehensive), pytest tests/e2e (full validation). Real example: 5,000 tests—unit: 3,500 tests in 5 min, integration: 1,000 tests in 10 min, e2e: 500 tests in 30 min. Without separation: Must run all 5,000 tests together (45 min wait for every commit). Flat/alphabetical/random don't enable speed-based selection.",
    },
    {
      id: 'tos-mc-5',
      question: 'What is the correct pytest naming convention for test files?',
      options: [
        'test_*.py or *_test.py',
        'test*.py (no underscore required)',
        '*_tests.py (plural "tests")',
        'unittest_*.py or pytest_*.py',
      ],
      correctAnswer: 0,
      explanation:
        'pytest discovers test files named test_*.py or *_test.py: test_calculator.py ✓ (recommended, more common). calculator_test.py ✓ (valid, less common in Python). test*.py ✗ (missing underscore, would match test.py but not testCalculator.py). *_tests.py ✗ (plural "tests" not standard, though some projects use it). unittest_/pytest_ prefix not part of convention. Consistency matters: Choose one pattern (test_* recommended) and use throughout project. Configure in pytest.ini: python_files = test_*.py. Also applies to functions (test_*) and classes (Test*).',
    },
  ];
