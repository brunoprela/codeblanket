/**
 * Multiple choice questions for Magic Methods (Dunder Methods) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const magicmethodsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is __init__?',
    options: [
      'A magic spell',
      'The constructor/initializer method',
      'A destructor',
      'A class variable',
    ],
    correctAnswer: 1,
    explanation:
      '__init__ is the initializer method called when creating a new instance of a class.',
  },
  {
    id: 'mc2',
    question: 'What does __str__ return?',
    options: [
      'A string representation for users',
      'The object type',
      'The object ID',
      'A boolean',
    ],
    correctAnswer: 0,
    explanation:
      '__str__ returns a user-friendly string representation, used by str() and print().',
  },
  {
    id: 'mc3',
    question: 'What is the difference between __str__ and __repr__?',
    options: [
      'No difference',
      '__str__ is user-friendly, __repr__ is developer-friendly/unambiguous',
      '__repr__ is faster',
      '__str__ is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      '__str__ for readable output (users), __repr__ for unambiguous representation (developers/debugging).',
  },
  {
    id: 'mc4',
    question: 'What does __len__ allow?',
    options: [
      'Calling len(obj)',
      'Comparing objects',
      'Adding objects',
      'Iterating over object',
    ],
    correctAnswer: 0,
    explanation:
      '__len__ allows len(obj) to work on custom objects, returning the "length" you define.',
  },
  {
    id: 'mc5',
    question: 'What are __enter__ and __exit__ used for?',
    options: [
      'Entering functions',
      'Context managers (with statement)',
      'Loops',
      'Error handling',
    ],
    correctAnswer: 1,
    explanation:
      '__enter__ and __exit__ enable objects to be used with the "with" statement as context managers.',
  },
];
