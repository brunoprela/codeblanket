/**
 * Python Object-Oriented Programming Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/python-oop/introduction';
import { classesobjectsSection } from '../sections/python-oop/classes-objects';
import { inheritanceSection } from '../sections/python-oop/inheritance';
import { dataclassesSection } from '../sections/python-oop/dataclasses';
import { propertiesSection } from '../sections/python-oop/properties';
import { compositionSection } from '../sections/python-oop/composition';
import { magicmethodsSection } from '../sections/python-oop/magic-methods';

// Import quizzes
import { introductionQuiz } from '../quizzes/python-oop/introduction';
import { classesobjectsQuiz } from '../quizzes/python-oop/classes-objects';
import { inheritanceQuiz } from '../quizzes/python-oop/inheritance';
import { dataclassesQuiz } from '../quizzes/python-oop/dataclasses';
import { propertiesQuiz } from '../quizzes/python-oop/properties';
import { compositionQuiz } from '../quizzes/python-oop/composition';
import { magicmethodsQuiz } from '../quizzes/python-oop/magic-methods';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/python-oop/introduction';
import { classesobjectsMultipleChoice } from '../multiple-choice/python-oop/classes-objects';
import { inheritanceMultipleChoice } from '../multiple-choice/python-oop/inheritance';
import { dataclassesMultipleChoice } from '../multiple-choice/python-oop/dataclasses';
import { propertiesMultipleChoice } from '../multiple-choice/python-oop/properties';
import { compositionMultipleChoice } from '../multiple-choice/python-oop/composition';
import { magicmethodsMultipleChoice } from '../multiple-choice/python-oop/magic-methods';

export const pythonOopModule: Module = {
  id: 'python-oop',
  title: 'Python Object-Oriented Programming',
  description:
    'Master object-oriented programming in Python including classes, inheritance, polymorphism, and design patterns.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🏗️',
  keyTakeaways: [
    'Classes bundle data and behavior—use __init__ to initialize instance attributes',
    'Inheritance models "is-a" relationships—use super() to call parent methods',
    'Polymorphism allows using objects of different types through common interface',
    'Composition ("has-a") is often better than inheritance for flexibility',
    'Abstract base classes define interfaces that subclasses must implement',
    'Dataclasses reduce boilerplate for data-focused classes—use field(default_factory) for mutables',
    'Properties provide controlled attribute access with validation and computed values',
    'Prefer composition over inheritance—build complex objects from simple, reusable components',
    'Magic methods make objects Pythonic—implement __repr__, __eq__, __lt__ for comparable objects',
    'Use @total_ordering to auto-generate comparison methods from __eq__ and __lt__',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...classesobjectsSection,
      quiz: classesobjectsQuiz,
      multipleChoice: classesobjectsMultipleChoice,
    },
    {
      ...inheritanceSection,
      quiz: inheritanceQuiz,
      multipleChoice: inheritanceMultipleChoice,
    },
    {
      ...dataclassesSection,
      quiz: dataclassesQuiz,
      multipleChoice: dataclassesMultipleChoice,
    },
    {
      ...propertiesSection,
      quiz: propertiesQuiz,
      multipleChoice: propertiesMultipleChoice,
    },
    {
      ...compositionSection,
      quiz: compositionQuiz,
      multipleChoice: compositionMultipleChoice,
    },
    {
      ...magicmethodsSection,
      quiz: magicmethodsQuiz,
      multipleChoice: magicmethodsMultipleChoice,
    },
  ],
};
