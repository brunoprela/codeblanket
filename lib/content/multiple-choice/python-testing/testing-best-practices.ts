import { MultipleChoiceQuestion } from '@/lib/types';

export const testingBestPracticesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tbp-mc-1',
    question: 'What does "test behavior, not implementation" mean?',
    options: [
      'Test all internal private methods thoroughly',
      'Test what the code does (outputs) not how it does it (internals)',
      'Never test implementation details under any circumstances',
      'Test only the fastest implementation',
    ],
    correctAnswer: 1,
    explanation:
      'Test WHAT not HOW: Behavior: Public API, inputs → outputs, user-visible effects. Implementation: Private methods, internal data structures, specific algorithms. Example: Testing add(2, 3) == 5 (behavior) vs testing internal _calculate() method (implementation). Why: Implementation tests break on refactoring. Behavior tests survive. Example: Switch from loop to list comprehension → implementation test breaks, behavior test passes. Test public methods, private methods covered implicitly. Exception: Extract complex private logic to public utility for testing.',
  },
  {
    id: 'tbp-mc-2',
    question: 'What is the Arrange-Act-Assert (AAA) pattern?',
    options: [
      'Arrange tests alphabetically, act on failures, assert correctness',
      'Arrange (setup), Act (execute), Assert (verify) - test structure pattern',
      'A pattern for organizing test files in directories',
      'Always Assert All results in every test',
    ],
    correctAnswer: 1,
    explanation:
      'AAA pattern structures tests: (1) Arrange: Setup test data. cart = ShoppingCart(); item = Product (price=29.99). (2) Act: Perform action. cart.add (item); total = cart.calculate_total(). (3) Assert: Verify result. assert total == 29.99. Benefits: Clear test flow, easy to read, separates concerns. Similar: Given-When-Then (BDD). Example: Given user with $100, When withdraw $50, Then balance $50. Not alphabetical, not file organization, not assert everything. Essential test structure for maintainability.',
  },
  {
    id: 'tbp-mc-3',
    question: 'Why is excessive mocking an anti-pattern?',
    options: [
      'Mocking is always bad and should never be used',
      'Over-mocking tests mocks instead of real behavior, missing actual bugs',
      'Mocks are slower than real objects',
      'Mocks require too much code to set up',
    ],
    correctAnswer: 1,
    explanation:
      'Excessive mocking tests mocks, not code: Example: Mock database, validator, email, logger → test only that mocks called, not actual logic. Miss: Integration bugs, SQL errors, validation logic issues. Better: Mock external dependencies only (email, external APIs), use real internal code (database, validators). Example: def test_user_service (db_session, mocker): mock_email = mocker.patch("send_email"); user = service.create (db_session) → Real DB + mocked email. Not never mock (external APIs need mocking), not slower (mocks faster), not setup (that\'s separate). Balance: Mock external, real internal.',
  },
  {
    id: 'tbp-mc-4',
    question: 'What is the test pyramid?',
    options: [
      'A 3D model of test execution order',
      'Many fast unit tests (base), fewer integration tests (middle), few E2E tests (top)',
      'A priority ranking of which tests to write first',
      'A visualization of test coverage percentages',
    ],
    correctAnswer: 1,
    explanation:
      'Test pyramid shows test distribution: Base (wide): 1000+ unit tests (fast, isolated, 80% of tests). Middle: 100+ integration tests (moderate speed, component interaction, 15%). Top (narrow): 10+ E2E tests (slow, full workflows, 5%). Why pyramid: Unit tests fast feedback (seconds), cheap to maintain. E2E tests slow (minutes), expensive. Balance: Many cheap fast tests, few expensive slow tests. Not execution order, not priority (all important), not coverage visualization. Professional standard for test suite structure.',
  },
  {
    id: 'tbp-mc-5',
    question: 'When should tests have multiple assertions?',
    options: [
      'Never, tests should only have one assertion',
      'When assertions verify related properties of the same behavior',
      'Always, to maximize coverage per test',
      'Only in integration tests, not unit tests',
    ],
    correctAnswer: 1,
    explanation:
      'Multiple assertions OK for related properties: Example: def test_user_creation(): user = User.create(...); assert user.id is not None; assert user.username == "alice"; assert user.created_at is not None. Related to same behavior (user creation). BAD: Unrelated assertions in one test. def test_everything(): assert add(2,3) == 5; assert User.count() == 0. Split into separate tests. Guideline: One logical concept per test, multiple assertions if verifying that concept. Not never (too restrictive), not always (loses focus), not integration-only (applies to all). Balance: Focus + completeness.',
  },
];
