/**
 * Multiple choice questions for Property Decorators Deep-Dive section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const propertiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main purpose of properties?',
    options: [
      'Make code run faster',
      'Allow methods to be accessed like attributes with validation',
      'Create private attributes',
      'Enable multiple inheritance',
    ],
    correctAnswer: 1,
    explanation:
      'Properties allow methods to be accessed like attributes while enabling validation, computed values, and controlled access.',
  },
  {
    id: 'mc2',
    question: 'How do you create a read-only property?',
    options: [
      'Use @readonly decorator',
      'Define only @property getter, no setter',
      'Set attribute to None',
      'Use const keyword',
    ],
    correctAnswer: 1,
    explanation:
      'Defining only the @property getter without a setter makes the property read-only. Attempting to set it will raise AttributeError.',
  },
  {
    id: 'mc3',
    question: 'When should you avoid using properties?',
    options: [
      'For simple attribute access',
      'For expensive or slow computations',
      'For validation',
      'For read-only attributes',
    ],
    correctAnswer: 1,
    explanation:
      'Avoid properties for expensive computations that take significant time. Properties should be fast since they look like attribute access. Use methods for slow operations.',
  },
  {
    id: 'mc4',
    question: 'What does @cached_property do (Python 3.8+)?',
    options: [
      'Makes property faster',
      'Computes property once and caches the result',
      'Makes property read-only',
      'Validates property value',
    ],
    correctAnswer: 1,
    explanation:
      '@cached_property computes the value once on first access and caches it, returning the cached value on subsequent accesses without recomputing.',
  },
  {
    id: 'mc5',
    question: 'What happens if you try to set a read-only property?',
    options: [
      'Value is silently ignored',
      'AttributeError is raised',
      'TypeError is raised',
      'Value is set successfully',
    ],
    correctAnswer: 1,
    explanation:
      'Attempting to set a read-only property (one without a setter) raises an AttributeError.',
  },
];
