/**
 * Multiple choice questions for Composition Over Inheritance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const compositionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Which relationship should use composition?',
    options: [
      'Dog is an Animal',
      'Car has an Engine',
      'Circle is a Shape',
      'Manager is an Employee',
    ],
    correctAnswer: 1,
    explanation:
      'Car HAS-AN Engine is a "has-a" relationship, perfect for composition. The others are "is-a" relationships suited for inheritance.',
  },
  {
    id: 'mc2',
    question: 'What is the main advantage of composition over inheritance?',
    options: [
      'Faster execution',
      'Less memory usage',
      'Greater flexibility and easier to change',
      'Simpler syntax',
    ],
    correctAnswer: 2,
    explanation:
      'Composition provides greater flexibility—you can swap components at runtime, avoid fragile base class problems, and more easily modify behavior.',
  },
  {
    id: 'mc3',
    question: 'What is delegation in the context of composition?',
    options: [
      'Creating subclasses',
      'Forwarding method calls to composed objects',
      'Multiple inheritance',
      'Private methods',
    ],
    correctAnswer: 1,
    explanation:
      'Delegation means forwarding method calls from the containing object to its composed objects, like car.start() calling self.engine.start().',
  },
  {
    id: 'mc4',
    question: 'When is inheritance appropriate?',
    options: [
      'When you have a "has-a" relationship',
      'When you want to reuse code',
      'When you have a true "is-a" relationship',
      'Always use inheritance',
    ],
    correctAnswer: 2,
    explanation:
      'Inheritance is appropriate for true "is-a" relationships where the child class can fully substitute for the parent (Liskov Substitution Principle).',
  },
  {
    id: 'mc5',
    question:
      'What problem does the Strategy Pattern (composition-based) solve?',
    options: [
      'Deep inheritance hierarchies',
      'Need for interchangeable algorithms without conditional logic',
      'Memory leaks',
      'Slow performance',
    ],
    correctAnswer: 1,
    explanation:
      'Strategy Pattern uses composition to provide interchangeable algorithms, avoiding conditional logic and making it easy to add new strategies without modifying existing code.',
  },
];
