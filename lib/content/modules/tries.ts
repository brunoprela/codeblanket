/**
 * Tries Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/tries/introduction';
import { implementationSection } from '../sections/tries/implementation';
import { patternsSection } from '../sections/tries/patterns';
import { advancedSection } from '../sections/tries/advanced';
import { complexitySection } from '../sections/tries/complexity';
import { templatesSection } from '../sections/tries/templates';
import { interviewSection } from '../sections/tries/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/tries/introduction';
import { implementationQuiz } from '../quizzes/tries/implementation';
import { patternsQuiz } from '../quizzes/tries/patterns';
import { advancedQuiz } from '../quizzes/tries/advanced';
import { complexityQuiz } from '../quizzes/tries/complexity';
import { templatesQuiz } from '../quizzes/tries/templates';
import { interviewQuiz } from '../quizzes/tries/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/tries/introduction';
import { implementationMultipleChoice } from '../multiple-choice/tries/implementation';
import { patternsMultipleChoice } from '../multiple-choice/tries/patterns';
import { advancedMultipleChoice } from '../multiple-choice/tries/advanced';
import { complexityMultipleChoice } from '../multiple-choice/tries/complexity';
import { templatesMultipleChoice } from '../multiple-choice/tries/templates';
import { interviewMultipleChoice } from '../multiple-choice/tries/interview';

export const triesModule: Module = {
  id: 'tries',
  title: 'Tries',
  description:
    'Master the prefix tree data structure for efficient string operations and searches.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🌲',
  keyTakeaways: [
    'Trie is a tree structure where each path represents a string prefix',
    'All operations (insert, search, prefix) are O(m) where m = string length',
    'Excellent for autocomplete, spell check, and prefix queries',
    'Space: O(ALPHABET_SIZE * m * n) worst case, O(m * n) best case',
    'Use hash map for flexible alphabet, array for fixed (faster but more space)',
    'Each node needs is_end_of_word flag to distinguish complete words from prefixes',
    'Can be extended with counts, wildcards, or other metadata per node',
    'More space-efficient than storing all prefixes separately',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...implementationSection,
      quiz: implementationQuiz,
      multipleChoice: implementationMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
    },
    {
      ...advancedSection,
      quiz: advancedQuiz,
      multipleChoice: advancedMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
    },
    {
      ...templatesSection,
      quiz: templatesQuiz,
      multipleChoice: templatesMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
