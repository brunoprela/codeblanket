/**
 * Multiple choice questions for Context Managers & Resource Management section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const contextmanagersMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When is __exit__ called in a context manager?',
    options: [
      'Only when the with block completes successfully',
      'Only when an exception occurs',
      'Always, whether an exception occurs or not',
      'Never, it must be called manually',
    ],
    correctAnswer: 2,
    explanation:
      '__exit__ is always called when leaving the with block, regardless of whether an exception occurred. This guarantees cleanup happens.',
  },
  {
    id: 'mc2',
    question: 'What does returning True from __exit__ do?',
    options: [
      'Indicates the context manager completed successfully',
      'Suppresses any exception that occurred in the with block',
      'Raises a new exception',
      'Forces the context manager to re-enter',
    ],
    correctAnswer: 1,
    explanation:
      'Returning True from __exit__ suppresses any exception that occurred in the with block. Return False (default) to let exceptions propagate normally.',
  },
  {
    id: 'mc3',
    question: 'Which module provides the @contextmanager decorator?',
    options: ['contextlib', 'functools', 'itertools', 'contextmanager'],
    correctAnswer: 0,
    explanation:
      'The contextlib module provides the @contextmanager decorator for creating simple context managers using generator functions.',
  },
  {
    id: 'mc4',
    question:
      'What is the main advantage of context managers over try/finally blocks?',
    options: [
      'Faster execution',
      'Less verbose and impossible to forget cleanup',
      'Automatic error recovery',
      'Parallel execution',
    ],
    correctAnswer: 1,
    explanation:
      'Context managers make cleanup code less verbose and guarantee it runs, making it impossible to forget cleanup. try/finally works but is verbose and error-prone.',
  },
  {
    id: 'mc5',
    question:
      'Can you use multiple context managers in a single with statement?',
    options: [
      'No, only one at a time',
      'Yes, separated by commas',
      'Only with special syntax',
      'Yes, but deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'You can use multiple context managers in a single with statement, separated by commas: with open(f1) as a, open(f2) as b:',
  },
];
