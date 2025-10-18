/**
 * Multiple choice questions for Testing & Debugging section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const testingdebuggingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the difference between unittest and pytest?',
    options: [
      'No difference',
      'pytest has simpler syntax and better features',
      'unittest is newer',
      'pytest only works with classes',
    ],
    correctAnswer: 1,
    explanation:
      'pytest offers simpler syntax (plain assert), fixtures, parametrization, and better error messages than unittest.',
  },
  {
    id: 'mc2',
    question: 'What does pytest fixture do?',
    options: [
      'Tests the code',
      'Provides reusable setup/teardown',
      'Fixes bugs',
      'Measures performance',
    ],
    correctAnswer: 1,
    explanation:
      'Fixtures provide reusable setup/teardown logic for tests, like database connections or test data.',
  },
  {
    id: 'mc3',
    question: 'What is TDD?',
    options: [
      'Test-Driven Development',
      'Test-Debug-Deploy',
      'Total Development Design',
      'Technical Design Document',
    ],
    correctAnswer: 0,
    explanation:
      'TDD (Test-Driven Development): write tests before code, ensuring testability and clear requirements.',
  },
  {
    id: 'mc4',
    question: 'What does pdb.set_trace() do?',
    options: [
      'Traces function calls',
      'Sets a breakpoint for debugging',
      'Measures execution time',
      'Logs errors',
    ],
    correctAnswer: 1,
    explanation:
      'pdb.set_trace() sets a breakpoint where code execution pauses, allowing interactive debugging.',
  },
  {
    id: 'mc5',
    question: 'What is pytest parametrize used for?',
    options: [
      'Measuring parameters',
      'Running same test with different inputs',
      'Setting up fixtures',
      'Debugging tests',
    ],
    correctAnswer: 1,
    explanation:
      '@pytest.mark.parametrize runs the same test with multiple input sets, reducing code duplication.',
  },
];
