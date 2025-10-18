/**
 * Multiple choice questions for Object-Oriented Programming in Python section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does encapsulation mean in OOP?',
    options: [
      'Using multiple classes',
      'Bundling data and methods that operate on that data',
      'Creating child classes from parent classes',
      'Using the same method name in different classes',
    ],
    correctAnswer: 1,
    explanation:
      'Encapsulation means bundling data (attributes) and methods that operate on that data together in a class, while hiding internal implementation details from outside code.',
  },
  {
    id: 'mc2',
    question: 'Which relationship should use inheritance?',
    options: [
      'A Car has an Engine',
      'A Student has a Backpack',
      'A Dog is an Animal',
      'A House has a Roof',
    ],
    correctAnswer: 2,
    explanation:
      'Inheritance represents "is-a" relationships. A Dog IS AN Animal is a true subtype relationship. The others are "has-a" relationships better modeled with composition.',
  },
  {
    id: 'mc3',
    question: 'What is polymorphism in OOP?',
    options: [
      'Having many classes',
      'Ability to use objects of different types through a common interface',
      'Creating multiple instances',
      'Hiding data from outside access',
    ],
    correctAnswer: 1,
    explanation:
      'Polymorphism allows objects of different types to be used through a common interface, enabling code that works with multiple types.',
  },
  {
    id: 'mc4',
    question: 'What is the main benefit of abstraction?',
    options: [
      'Faster code execution',
      'Hiding complexity and exposing only essential features',
      'Using less memory',
      'Creating more classes',
    ],
    correctAnswer: 1,
    explanation:
      'Abstraction hides complex implementation details and exposes only the essential features, making code easier to understand and use.',
  },
  {
    id: 'mc5',
    question: 'Which is an example of composition?',
    options: [
      'class Dog(Animal)',
      'class Car: self.engine = Engine()',
      'class Student(Person)',
      'class Circle(Shape)',
    ],
    correctAnswer: 1,
    explanation:
      'class Car with self.engine = Engine() is composition—a Car HAS AN Engine. The others show inheritance (is-a relationships).',
  },
];
