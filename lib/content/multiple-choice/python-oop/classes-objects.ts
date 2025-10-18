/**
 * Multiple choice questions for Classes and Objects section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const classesobjectsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the purpose of the __init__ method?',
    options: [
      'To delete an object',
      "To initialize an object's attributes when it is created",
      'To compare two objects',
      'To convert an object to a string',
    ],
    correctAnswer: 1,
    explanation:
      "__init__ is the constructor method called when creating a new instance. It initializes the object's attributes with values passed as arguments.",
  },
  {
    id: 'mc2',
    question: 'What does @property decorator do?',
    options: [
      'Makes a method private',
      'Allows a method to be accessed like an attribute',
      'Creates a class method',
      'Makes an attribute immutable',
    ],
    correctAnswer: 1,
    explanation:
      '@property decorator allows a method to be accessed like an attribute without parentheses, enabling computed values, validation, and controlled access.',
  },
  {
    id: 'mc3',
    question: 'What is the purpose of the __str__ method?',
    options: [
      'To convert the object to an integer',
      'To define how an object is represented as a string',
      'To create a new object',
      'To delete an object',
    ],
    correctAnswer: 1,
    explanation:
      '__str__ defines how an object should be represented as a human-readable string, used by print() and str().',
  },
  {
    id: 'mc4',
    question: 'What does @classmethod receive as its first parameter?',
    options: [
      'self (the instance)',
      'cls (the class)',
      'No parameter',
      'The parent class',
    ],
    correctAnswer: 1,
    explanation:
      '@classmethod receives cls (the class) as the first parameter, allowing it to access and modify class state.',
  },
  {
    id: 'mc5',
    question: 'Which is true about class attributes?',
    options: [
      'Each instance has its own copy',
      'They are shared by all instances of the class',
      'They cannot be modified',
      'They must be private',
    ],
    correctAnswer: 1,
    explanation:
      'Class attributes are shared by all instances of the class. Modifying a class attribute affects all instances.',
  },
];
