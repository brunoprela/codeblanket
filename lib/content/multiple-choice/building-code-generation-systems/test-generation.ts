/**
 * Multiple choice questions for Test Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const testgenerationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcgs-testgen-mc-1',
    question:
      'What is the most important principle when generating unit tests?',
    options: [
      'Achieve 100% code coverage',
      'Test behavior and edge cases, not just happy path',
      'Generate as many tests as possible',
      'Only test public methods',
    ],
    correctAnswer: 1,
    explanation:
      'Tests should verify behavior (inputs → outputs) across happy path AND edge cases (empty inputs, invalid inputs, boundary conditions, errors). Coverage is metric, not goal.',
  },
  {
    id: 'bcgs-testgen-mc-2',
    question: 'How should LLM-generated tests be structured?',
    options: [
      'All in one test function',
      'Following Arrange-Act-Assert (AAA) pattern',
      'Random order is fine',
      'No structure needed',
    ],
    correctAnswer: 1,
    explanation:
      'Arrange-Act-Assert (AAA) pattern: 1) Arrange: set up test data. 2) Act: call function being tested. 3) Assert: verify output/behavior. Clear structure makes tests readable.',
  },
  {
    id: 'bcgs-testgen-mc-3',
    question: 'What should be mocked in unit tests?',
    options: [
      'Nothing, test everything together',
      'External dependencies (APIs, databases, file system)',
      'All functions',
      'Only the function being tested',
    ],
    correctAnswer: 1,
    explanation:
      'Mock external dependencies (APIs, databases, file system) to isolate the unit being tested. Unit tests should be fast, deterministic, and not depend on external state.',
  },
  {
    id: 'bcgs-testgen-mc-4',
    question: 'How should generated tests handle edge cases?',
    options: [
      'Ignore them to keep tests simple',
      "Document them but don't test",
      'Create specific test cases for each edge case',
      'Only test one edge case',
    ],
    correctAnswer: 2,
    explanation:
      'Create specific test cases for each edge case: empty inputs, null/undefined, boundary values, invalid inputs, errors. Edge cases are where bugs hide.',
  },
  {
    id: 'bcgs-testgen-mc-5',
    question:
      'What information should the LLM receive to generate quality tests?',
    options: [
      'Only function signature',
      'Function code, docstrings, type hints, and existing test patterns',
      'Just the function name',
      'Only the file containing the function',
    ],
    correctAnswer: 1,
    explanation:
      'Provide function code (to understand logic), docstrings (to understand intent), type hints (to understand data), and existing test patterns (to match style). More context = better tests.',
  },
];
