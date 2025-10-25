import { MultipleChoiceQuestion } from '@/lib/types';

export const parametrizedTestsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pt-mc-1',
    question:
      'What is the main benefit of parametrized tests over writing multiple similar test functions?',
    options: [
      'Parametrized tests run faster than individual test functions',
      'Parametrized tests eliminate code duplication by testing same logic with different inputs',
      'Parametrized tests automatically generate test data without manual specification',
      'Parametrized tests can only be used with fixtures, not with regular functions',
    ],
    correctAnswer: 1,
    explanation:
      'Parametrized tests eliminate code duplication: Write one test function that runs multiple times with different parameters instead of writing N nearly-identical test functions. Example: @pytest.mark.parametrize("a,b,expected", [(1,2,3), (2,3,5)]) def test_add (a,b,expected): assert add (a,b)==expected → 1 test runs 2× instead of 2 separate tests. Benefit: 65% less code, easier to maintain, easier to add new cases (1 line vs new function). Not faster: Same execution time. Doesn\'t auto-generate data: Must provide parameters. Works with or without fixtures.',
  },
  {
    id: 'pt-mc-2',
    question:
      'When using multiple @pytest.mark.parametrize decorators on the same test, how many tests are executed?',
    options: [
      "Only the last decorator's parameters are used",
      'Parameters are merged into a single set',
      'The product (multiplication) of all parameter sets (matrix expansion)',
      'pytest raises an error—only one parametrize decorator allowed',
    ],
    correctAnswer: 2,
    explanation:
      'Multiple parametrize decorators create matrix expansion (Cartesian product): @pytest.mark.parametrize("a", [1,2]) @pytest.mark.parametrize("b", [3,4]) → 2×2=4 tests: (1,3), (1,4), (2,3), (2,4). Example: 3 currencies × 4 HTTP methods × 5 endpoints = 60 tests from 3 decorators. Not sequential: Both decorators apply. Not error: Multiple decorators fully supported. Use case: Test all combinations of independent variables. Warning: Can explode quickly (10×10×10 = 1,000 tests).',
  },
  {
    id: 'pt-mc-3',
    question: 'What does indirect=True do in parametrize?',
    options: [
      'Runs tests indirectly through a separate test runner',
      'Passes the parameter through a fixture instead of directly to the test function',
      'Skips the test and marks it for later execution',
      'Allows parametrization to work with async test functions',
    ],
    correctAnswer: 1,
    explanation:
      'indirect=True passes parameter through fixture: @pytest.fixture def user (request): name=request.param; return User (name); @pytest.mark.parametrize("user", ["Alice","Bob"], indirect=True) def test_user (user): ... → Parameter goes to fixture first (creates User object), test receives User object (not string). Without indirect: test receives raw parameter ("Alice" string). Use case: Complex setup needed for each parameter (create database, authenticate user, etc.). Partial indirect: indirect=["param1"] → only param1 goes through fixture. Not for skipping/async: Different features.',
  },
  {
    id: 'pt-mc-4',
    question: 'What is the purpose of the ids parameter in parametrize?',
    options: [
      'To assign unique database IDs to each test case',
      'To provide readable test names in the output instead of parameter values',
      'To identify which parameters should be treated as indirect',
      'To specify the order in which parametrized tests should run',
    ],
    correctAnswer: 1,
    explanation:
      'ids parameter provides readable test names: Without: test_payment[100.0-USD] (uses parameter values). With: ids=["small-payment","large-payment"] → test_payment[small-payment] (readable). Dynamic: ids=lambda x: f"amount-{x[0]}" generates IDs from parameters. Benefits: Easier to identify failing tests, better test reports, clearer CI output. Not for: database IDs, indirect specification, or execution order. Example: 50 test cases with cryptic parameter values → use IDs for clarity. Best practice: Always use ids for parameters that don\'t self-document (numbers, tuples).',
  },
  {
    id: 'pt-mc-5',
    question:
      'How can you skip a specific parameter set in a parametrized test?',
    options: [
      'Use if statement inside the test to return early',
      'Remove that parameter set from the list',
      'Use pytest.param with marks=pytest.mark.skip',
      'Parametrized tests cannot skip individual parameter sets',
    ],
    correctAnswer: 2,
    explanation:
      'Use pytest.param with marks=pytest.mark.skip: @pytest.mark.parametrize("value", [1, 2, pytest.param(3, marks=pytest.mark.skip (reason="Known bug")), 4]) → Tests run for 1,2,4 but skip 3. Also: marks=pytest.mark.xfail (expected to fail), marks=pytest.mark.skipif (conditional). Alternative: Can remove from list, but marks are better (documents why skipped, shows in test output). Not if statement: Runs test (wastes time), doesn\'t show as skipped in report. Parametrized tests CAN skip individual sets with pytest.param. Output: test[3] SKIPPED (Known bug).',
  },
];
