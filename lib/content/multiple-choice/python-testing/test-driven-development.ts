import { MultipleChoiceQuestion } from '@/lib/types';

export const testDrivenDevelopmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tdd-mc-1',
    question: 'What are the three steps of TDD in order?',
    options: [
      'Write code, write test, refactor',
      'Red (write failing test), Green (make it pass), Refactor',
      'Design, implement, test',
      'Test, debug, deploy',
    ],
    correctAnswer: 1,
    explanation:
      "TDD cycle: Red-Green-Refactor. (1) RED: Write failing test first. Example: def test_add(): assert Calculator().add(2, 3) == 5 → FAILS (no code). (2) GREEN: Write minimal code to pass. class Calculator: def add (self, a, b): return a + b → PASSES. (3) REFACTOR: Improve code while keeping tests green. Add type hints, docstrings, optimize. Repeat for next feature. Not code-first (that's test-after), not design phase (tests ARE design), not debug/deploy (that's later).",
  },
  {
    id: 'tdd-mc-2',
    question: 'In TDD, what does "Red" mean?',
    options: [
      'The test is too complex and needs simplification',
      "The test fails because the code doesn't exist yet",
      'There is a syntax error in the test',
      'The code coverage is below threshold',
    ],
    correctAnswer: 1,
    explanation:
      'RED means failing test (expected): Write test first before code exists. Example: def test_divide(): assert calc.divide(10, 2) == 5 → FAILS (ImportError or AttributeError). This is GOOD in TDD—confirms test will catch bugs. Then write code (GREEN), test passes. Red ensures test actually tests something. If test passes immediately ("false positive"), test is broken. Red is intentional, not error or coverage issue.',
  },
  {
    id: 'tdd-mc-3',
    question: 'What should you do in the GREEN step of TDD?',
    options: [
      'Write the most elegant, optimized solution',
      'Write the minimal code needed to make the test pass',
      'Write all related features at once',
      'Refactor existing code',
    ],
    correctAnswer: 1,
    explanation:
      "GREEN step: Write minimal code to pass test. Example: Test expects add(2, 3) == 5. Minimal: return a + b. Don't: Add validation, optimization, edge cases yet. Why minimal: (1) Fast feedback (get to green quickly), (2) Avoid over-engineering, (3) Next test will drive next requirement. Refactoring comes in REFACTOR step (after green). Not elegant (that's refactor), not all features (one test at a time), not refactoring (that's step 3).",
  },
  {
    id: 'tdd-mc-4',
    question: 'How should you handle a production bug using TDD?',
    options: [
      'Fix the bug immediately, then write a test',
      'Write a failing test that reproduces the bug, then fix it',
      'Remove the feature causing the bug',
      'Add more logging to find the root cause',
    ],
    correctAnswer: 1,
    explanation:
      "TDD bug fix: (1) Write failing test reproducing bug. Example: def test_divide_by_zero(): with pytest.raises(ValueError): calc.divide(10, 0) → FAILS (bug: no error raised). (2) Fix code: if b == 0: raise ValueError. (3) Test PASSES. Benefits: Confirms bug exists, prevents regression (test catches if bug returns), documents bug. Not fix-first (can't verify fix works), not remove feature, not just logging (test prevents recurrence).",
  },
  {
    id: 'tdd-mc-5',
    question: 'What is a key advantage of TDD over writing tests after code?',
    options: [
      'TDD requires fewer tests overall',
      'TDD forces simple, testable design from the start',
      'TDD tests run faster than test-after tests',
      'TDD eliminates the need for integration tests',
    ],
    correctAnswer: 1,
    explanation:
      'TDD forces testable design: Writing tests first makes you think "How do I test this?" → simple interfaces. Example: TDD: add (a, b) → easy to test. Test-after: calc = Calculator (db, cache, logger); calc.add (a, b) → hard to test (many dependencies). TDD prevents over-complicated designs (if hard to test, redesign). Test-after often produces untestable code → skip tests. Not fewer tests (same coverage), not faster (same tests), doesn\'t eliminate integration tests (still needed).',
  },
];
