import { MultipleChoiceQuestion } from '@/lib/types';

export const testingFundamentalsPytestBasicsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tfpb-mc-1',
      question:
        'According to the testing pyramid, what percentage of your test suite should be unit tests?',
      options: [
        '5-10% (unit tests are slow and should be minimized)',
        '25-35% (equal distribution across all test types)',
        '60-75% (majority should be fast, isolated unit tests)',
        '95-100% (only unit tests are necessary)',
      ],
      correctAnswer: 2,
      explanation:
        'The testing pyramid recommends 60-75% unit tests because they are: Fast (<100ms), cheap to maintain, reliable (no external dependencies), catch most bugs early. Unit tests should form the foundation. Integration tests (20-30%) test component interactions. E2E tests (5-10%) are slow and brittle but test critical user journeys. Real-world example: 1000-test suite ideally has ~700 unit tests (run in 1 minute), ~250 integration tests (run in 5 minutes), ~50 E2E tests (run in 10 minutes). This balance provides comprehensive coverage with fast feedback.',
    },
    {
      id: 'tfpb-mc-2',
      question:
        'What is the main advantage of pytest assert statements over unittest assertions?',
      options: [
        'pytest asserts are faster to execute than unittest assertions',
        'pytest uses plain assert with introspection for detailed errors, unittest uses verbose methods',
        'pytest asserts can test more complex conditions than unittest',
        'pytest asserts are required by Python 3.10+ while unittest is deprecated',
      ],
      correctAnswer: 1,
      explanation:
        'pytest uses plain Python assert statements with introspection: assert result == 5 gives detailed output showing actual vs expected values. unittest requires verbose methods: self.assertEqual(result, 5). pytest output example: "assert 3 == 5" shows "where 3 = calc.add(1, 2)". This introspection makes debugging much easier. No performance difference. Both can test same conditions. Neither is deprecated—pytest is just more Pythonic and readable. Example: pytest assertion "assert user.name == \'Alice\'" vs unittest "self.assertEqual(user.name, \'Alice\')"—pytest is 40% less code and more natural Python.',
    },
    {
      id: 'tfpb-mc-3',
      question: 'What does the AAA pattern in testing stand for?',
      options: [
        'Assert, Act, Arrange',
        'Arrange, Assert, Act',
        'Arrange, Act, Assert',
        'Act, Arrange, Assert',
      ],
      correctAnswer: 2,
      explanation:
        'AAA = Arrange, Act, Assert (in that order): (1) Arrange: Set up test data and preconditions (create objects, mock dependencies). (2) Act: Execute the code being tested (call the function/method). (3) Assert: Verify the results match expectations (check return values, side effects, exceptions). Example: def test_payment(): # Arrange payment = Payment(amount=100.0) # Act result = process_payment(payment) # Assert assert result is True. This pattern makes tests readable and maintainable by separating concerns. Also called Given-When-Then in BDD. Following AAA makes tests self-documenting.',
    },
    {
      id: 'tfpb-mc-4',
      question: 'Why is test independence critical in a large test suite?',
      options: [
        'Independent tests run faster than dependent tests',
        'Independent tests can run in any order and in parallel without affecting results',
        'Independent tests require less code than dependent tests',
        'Independent tests automatically have better coverage',
      ],
      correctAnswer: 1,
      explanation:
        "Test independence ensures tests can run in any order and in parallel: (1) Parallel execution: pytest -n 8 runs 8 tests simultaneously—only works if tests don't share state. (2) Isolation: One test failure doesn't cascade to others. (3) Debugging: Can run single test without running entire suite first. (4) Randomization: pytest-randomly shuffles tests to catch hidden dependencies. Anti-pattern: test_create_user() creates global user, test_delete_user() depends on it—fails if run alone. Best practice: Each test creates its own test data. Real-world impact: 1000 tests × 0.5s = 8 minutes serial, 1 minute parallel (8 cores). Dependent tests cannot parallelize—lose 7 minutes per run.",
    },
    {
      id: 'tfpb-mc-5',
      question:
        'What is the primary purpose of the conftest.py file in pytest?',
      options: [
        'To configure pytest command-line options',
        'To define shared fixtures and test utilities available to all test files',
        'To list all test files that should be executed',
        'To store test coverage configuration',
      ],
      correctAnswer: 1,
      explanation:
        'conftest.py defines shared fixtures available to all tests in that directory and subdirectories: Fixtures defined here are automatically discovered (no import needed). Example: @pytest.fixture in conftest.py → available to all test files. Scope: Root conftest.py affects entire project, tests/unit/conftest.py affects only unit tests. Best practices: Database fixtures (db_session), API client fixtures, test data factories. NOT for: pytest.ini handles command-line options, .coveragerc handles coverage config. Real project: conftest.py at root with database/Redis fixtures (100 lines), each test file uses them without import. Multiple conftest.py files can exist at different levels (fixtures compose/override).',
    },
  ];
