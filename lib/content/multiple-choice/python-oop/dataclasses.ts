/**
 * Multiple choice questions for Dataclasses for Structured Data section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dataclassesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does the @dataclass decorator automatically generate?',
    options: [
      'Only __init__',
      '__init__, __repr__, and __eq__',
      'Only __str__',
      'All dunder methods',
    ],
    correctAnswer: 1,
    explanation:
      '@dataclass automatically generates __init__, __repr__, __eq__, and optionally __hash__ and __order__ methods based on the fields.',
  },
  {
    id: 'mc2',
    question:
      'Why must you use field (default_factory=list) for mutable defaults?',
    options: [
      "It\'s faster",
      'To avoid sharing mutable objects between instances',
      'Required by Python syntax',
      'To make the list immutable',
    ],
    correctAnswer: 1,
    explanation:
      'field (default_factory=list) creates a new list for each instance, preventing the shared mutable default argument gotcha where all instances would share the same list.',
  },
  {
    id: 'mc3',
    question: 'What does frozen=True do in a dataclass?',
    options: [
      'Makes the class run faster',
      'Makes instances immutable',
      'Freezes the class at creation time',
      'Prevents inheritance',
    ],
    correctAnswer: 1,
    explanation:
      'frozen=True makes dataclass instances immutable—you cannot modify attributes after creation, similar to tuples. This also makes them hashable.',
  },
  {
    id: 'mc4',
    question: 'When is __post_init__ called?',
    options: [
      'Before __init__',
      'After __init__, for additional processing',
      'When the object is deleted',
      'Only on first access',
    ],
    correctAnswer: 1,
    explanation:
      '__post_init__ is called automatically after __init__ completes, allowing you to perform validation, compute derived values, or other initialization logic.',
  },
  {
    id: 'mc5',
    question: 'What is the main advantage of dataclasses over regular classes?',
    options: [
      'Faster execution',
      'Less memory usage',
      'Reduced boilerplate code',
      'Better inheritance',
    ],
    correctAnswer: 2,
    explanation:
      'The main advantage is reduced boilerplate—dataclasses automatically generate __init__, __repr__, __eq__ and other methods, saving you from writing repetitive code.',
  },
];
