/**
 * Arrays & Hashing Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/arrays-hashing/introduction';
import { arraysSection } from '../sections/arrays-hashing/arrays';
import { hashingSection } from '../sections/arrays-hashing/hashing';
import { complexitySection } from '../sections/arrays-hashing/complexity';
import { patternsSection } from '../sections/arrays-hashing/patterns';
import { advancedSection } from '../sections/arrays-hashing/advanced';
import { twosumpatternsSection } from '../sections/arrays-hashing/two-sum-patterns';
import { interviewSection } from '../sections/arrays-hashing/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/arrays-hashing/introduction';
import { arraysQuiz } from '../quizzes/arrays-hashing/arrays';
import { hashingQuiz } from '../quizzes/arrays-hashing/hashing';
import { complexityQuiz } from '../quizzes/arrays-hashing/complexity';
import { patternsQuiz } from '../quizzes/arrays-hashing/patterns';
import { advancedQuiz } from '../quizzes/arrays-hashing/advanced';
import { twosumpatternsQuiz } from '../quizzes/arrays-hashing/two-sum-patterns';
import { interviewQuiz } from '../quizzes/arrays-hashing/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/arrays-hashing/introduction';
import { arraysMultipleChoice } from '../multiple-choice/arrays-hashing/arrays';
import { hashingMultipleChoice } from '../multiple-choice/arrays-hashing/hashing';
import { complexityMultipleChoice } from '../multiple-choice/arrays-hashing/complexity';
import { patternsMultipleChoice } from '../multiple-choice/arrays-hashing/patterns';
import { advancedMultipleChoice } from '../multiple-choice/arrays-hashing/advanced';
import { twosumpatternsMultipleChoice } from '../multiple-choice/arrays-hashing/two-sum-patterns';
import { interviewMultipleChoice } from '../multiple-choice/arrays-hashing/interview';

export const arraysHashingModule: Module = {
  id: 'arrays-hashing',
  title: 'Arrays & Hashing',
  description:
    'Master the fundamentals of array manipulation and hash table techniques for optimal performance.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔢',
  keyTakeaways: [
    'Arrays provide O(1) indexed access but O(n) search; use binary search for O(log n) on sorted arrays',
    'Hash tables provide O(1) average lookup, insert, delete - essential for optimization',
    'Frequency counting: use Counter or dict to count occurrences in O(n)',
    'Two Sum pattern: store complements in hash table for O(n) solution',
    'Grouping pattern: use defaultdict to partition elements by key',
    'Deduplication: use set for O(n) duplicate detection vs O(n²) brute force',
    'Space-time tradeoff: hash tables use O(n) space but often reduce time from O(n²) to O(n)',
    'Choose dict for key-value, set for membership, Counter for frequencies, defaultdict for auto-initialization',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...arraysSection,
      quiz: arraysQuiz,
      multipleChoice: arraysMultipleChoice,
    },
    {
      ...hashingSection,
      quiz: hashingQuiz,
      multipleChoice: hashingMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
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
      ...twosumpatternsSection,
      quiz: twosumpatternsQuiz,
      multipleChoice: twosumpatternsMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
