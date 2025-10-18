/**
 * Multiple choice questions for Metaclasses & Class Creation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const metaclassesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the default metaclass in Python?',
    options: ['object', 'type', 'class', 'meta'],
    correctAnswer: 1,
    explanation:
      'type is the default metaclass in Python. All classes (unless specified otherwise) are instances of type.',
  },
  {
    id: 'mc2',
    question: 'When should you prefer class decorators over metaclasses?',
    options: [
      'When you need to modify the class structure',
      'When you need simpler, more readable class modification',
      'When you need to control all subclasses',
      'When implementing an ORM',
    ],
    correctAnswer: 1,
    explanation:
      'Class decorators are simpler and more readable than metaclasses for most use cases. Use metaclasses only when you need to control subclass creation or modify the class at a structural level.',
  },
  {
    id: 'mc3',
    question:
      'Which method in a metaclass is called first during class creation?',
    options: ['__init__', '__new__', '__call__', '__create__'],
    correctAnswer: 1,
    explanation:
      '__new__ is called first in a metaclass to create the class object before __init__ initializes it.',
  },
  {
    id: 'mc4',
    question: 'What is a common use case for metaclasses?',
    options: [
      'Sorting lists',
      'ORM implementations like Django models',
      'File I/O',
      'String manipulation',
    ],
    correctAnswer: 1,
    explanation:
      'ORMs like Django use metaclasses to transform class definitions into database schemas, automatically creating fields and methods.',
  },
  {
    id: 'mc5',
    question: 'What is __init_subclass__ used for?',
    options: [
      'Initializing object instances',
      'A simpler alternative to metaclasses for controlling subclass creation',
      'Defining class variables',
      'Creating abstract methods',
    ],
    correctAnswer: 1,
    explanation:
      '__init_subclass__ (Python 3.6+) is a simpler alternative to metaclasses for customizing subclass creation, without needing a metaclass.',
  },
];
