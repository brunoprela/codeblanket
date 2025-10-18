/**
 * Multiple choice questions for Advanced Python: Beyond the Basics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does the yield keyword do in Python?',
    options: [
      'It returns a value and exits the function',
      'It produces a value and pauses the function, maintaining its state',
      'It creates a new thread',
      'It raises an exception',
    ],
    correctAnswer: 1,
    explanation:
      'yield produces a value and pauses the generator function, maintaining its state (local variables, instruction pointer) until the next value is requested. This allows for lazy evaluation and memory-efficient iteration.',
  },
  {
    id: 'mc2',
    question: 'What is the main advantage of using decorators?',
    options: [
      'Faster execution speed',
      'Reduced memory usage',
      'Code reusability and separation of concerns',
      'Automatic error handling',
    ],
    correctAnswer: 2,
    explanation:
      'Decorators allow you to reuse functionality across multiple functions and separate cross-cutting concerns (like logging, authentication) from business logic, following the DRY principle.',
  },
  {
    id: 'mc3',
    question:
      'Which methods must a context manager implement to work with the "with" statement?',
    options: [
      '__init__ and __del__',
      '__enter__ and __exit__',
      '__start__ and __end__',
      '__open__ and __close__',
    ],
    correctAnswer: 1,
    explanation:
      'Context managers must implement __enter__ (called when entering the with block) and __exit__ (called when leaving, even if an exception occurred).',
  },
  {
    id: 'mc4',
    question: 'When should you use a generator instead of returning a list?',
    options: [
      'When you need random access to elements',
      'When processing large datasets that might not fit in memory',
      'When you need to sort the results',
      'When you need to access elements multiple times',
    ],
    correctAnswer: 1,
    explanation:
      'Generators are ideal for large datasets because they produce values on-demand (lazy evaluation) rather than creating the entire list in memory at once. This makes them memory-efficient.',
  },
  {
    id: 'mc5',
    question: 'What is a metaclass in Python?',
    options: [
      'A class that inherits from multiple parents',
      'A class of classes that controls class creation',
      'A class with only class methods',
      'An abstract base class',
    ],
    correctAnswer: 1,
    explanation:
      'A metaclass is a class of classes - it controls how classes are created and behave, similar to how classes control how instances are created.',
  },
];
