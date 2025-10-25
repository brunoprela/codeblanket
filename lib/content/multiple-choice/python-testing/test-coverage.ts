import { MultipleChoiceQuestion } from '@/lib/types';

export const testCoverageMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tc-mc-1',
    question: 'What does 80% test coverage mean?',
    options: [
      '80% of tests pass',
      '80% of code lines are executed during tests',
      '80% of bugs are caught by tests',
      '80% of features are tested',
    ],
    correctAnswer: 1,
    explanation:
      "80% coverage means 80% of code lines executed during tests: Example: 100 lines total, tests execute 80 lines = 80% coverage. 20 lines never run during tests (untested). Coverage measures execution, not: test pass rate (that's test results), bug detection (no correlation), or feature completion. High coverage doesn't guarantee quality—can have 100% coverage with no assertions. Use coverage to find untested code, then write meaningful tests for it.",
  },
  {
    id: 'tc-mc-2',
    question: 'What is branch coverage?',
    options: [
      'Coverage of different Git branches',
      'Coverage of all decision paths (if/else, try/except)',
      'Coverage of all function branches in code',
      'Coverage of all test branches in test suite',
    ],
    correctAnswer: 1,
    explanation:
      'Branch coverage tests all decision paths: if amount > 0: success else: error → Need tests for both amount > 0 (True) and amount <= 0 (False). Line coverage: 100% if both lines run once. Branch coverage: 100% if both branches tested. Example: pytest --cov-branch shows branch coverage. Why important: Line coverage misses untested paths. if error: raise Exception might have 100% line coverage but never test the error case. Branch coverage catches this.',
  },
  {
    id: 'tc-mc-3',
    question: 'What does --cov-fail-under=80 do?',
    options: [
      'Skips tests with coverage under 80%',
      'Fails pytest if coverage drops below 80%',
      'Only reports files with coverage under 80%',
      'Sets the warning threshold to 80%',
    ],
    correctAnswer: 1,
    explanation:
      '--cov-fail-under=80 fails pytest if coverage < 80%: pytest --cov=src --cov-fail-under=80 → exit code 1 if 79%, exit code 0 if 80%+. Use in CI: Prevent merges that reduce coverage, maintain quality standards. Example: pytest.ini: [pytest] addopts = --cov-fail-under=80. Not for: skipping tests, filtering reports, or warnings. Essential CI/CD tool: Enforces coverage requirements, prevents regression.',
  },
  {
    id: 'tc-mc-4',
    question: 'Why might 100% coverage give false confidence?',
    options: [
      'Coverage measures execution, not assertion quality',
      '100% coverage is impossible to achieve',
      'Coverage tools are unreliable',
      '100% coverage makes tests too slow',
    ],
    correctAnswer: 0,
    explanation:
      'Coverage measures execution, not correctness: def test_add(): add(2, 3) # NO assertion! → 100% coverage but no verification. False confidence: All code runs but nothing checked. Better: 80% coverage with assert result == 5 than 100% with no assertions. Coverage shows what code runs, not whether it works correctly. Good tests: Execute code AND verify behavior. Not impossible (can reach 100%), tools reliable, speed unrelated.',
  },
  {
    id: 'tc-mc-5',
    question: 'What is the purpose of # pragma: no cover?',
    options: [
      'Marks code that should not be executed',
      'Excludes code from coverage reports',
      'Disables coverage for entire file',
      'Marks code that requires 100% coverage',
    ],
    correctAnswer: 1,
    explanation:
      '# pragma: no cover excludes code from coverage reports: def debug_func(): # pragma: no cover → not counted in coverage. Use for: Debug code, development helpers, unreachable code, trivial methods. Example: if __name__ == "__main__": # pragma: no cover → exclude scripts from coverage. Not for: preventing execution, disabling entire files (.coveragerc for that), or requiring coverage. Reduces noise in coverage reports by excluding non-critical code.',
  },
];
