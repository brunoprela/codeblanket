/**
 * Multiple choice questions for Inheritance and Polymorphism section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const inheritanceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does super() do?',
    options: [
      'Creates a superclass',
      "Calls the parent class's method",
      'Makes a method private',
      'Creates multiple inheritance',
    ],
    correctAnswer: 1,
    explanation:
      'super() returns a temporary object that allows you to call methods of the parent class, enabling proper method chaining in inheritance hierarchies.',
  },
  {
    id: 'mc2',
    question: 'What is polymorphism?',
    options: [
      'Having multiple classes',
      'Using the same interface for different data types',
      'Inheriting from multiple parents',
      'Hiding implementation details',
    ],
    correctAnswer: 1,
    explanation:
      'Polymorphism means using the same interface (method name) for different data types. Different classes can implement the same method in their own way.',
  },
  {
    id: 'mc3',
    question: 'What happens if you try to instantiate an abstract base class?',
    options: [
      'It works normally',
      'TypeError is raised',
      'Returns None',
      'Creates an empty object',
    ],
    correctAnswer: 1,
    explanation:
      'Python raises a TypeError if you try to instantiate an abstract base class that has abstract methods. You must create a concrete subclass that implements all abstract methods.',
  },
  {
    id: 'mc4',
    question:
      'In class Child(Parent1, Parent2), what is the order of parent class checking?',
    options: [
      'Parent2, then Parent1',
      'Parent1, then Parent2',
      'Random order',
      'Only checks Parent1',
    ],
    correctAnswer: 1,
    explanation:
      'Python checks parent classes from left to right: Parent1, then Parent2. This is part of the Method Resolution Order (MRO).',
  },
  {
    id: 'mc5',
    question: 'What does the @abstractmethod decorator do?',
    options: [
      'Makes a method private',
      'Marks a method that must be implemented by subclasses',
      'Makes a method faster',
      'Converts a method to a class method',
    ],
    correctAnswer: 1,
    explanation:
      '@abstractmethod marks a method that subclasses must implement. Classes with abstract methods cannot be instantiated.',
  },
];
