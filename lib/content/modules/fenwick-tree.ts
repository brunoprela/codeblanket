/**
 * Fenwick Tree (Binary Indexed Tree) Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/fenwick-tree/introduction';
import { structureSection } from '../sections/fenwick-tree/structure';
import { operationsSection } from '../sections/fenwick-tree/operations';
import { advancedSection } from '../sections/fenwick-tree/advanced';
import { comparisonSection } from '../sections/fenwick-tree/comparison';
import { interviewstrategySection } from '../sections/fenwick-tree/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/fenwick-tree/introduction';
import { structureQuiz } from '../quizzes/fenwick-tree/structure';
import { operationsQuiz } from '../quizzes/fenwick-tree/operations';
import { advancedQuiz } from '../quizzes/fenwick-tree/advanced';
import { comparisonQuiz } from '../quizzes/fenwick-tree/comparison';
import { interviewstrategyQuiz } from '../quizzes/fenwick-tree/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/fenwick-tree/introduction';
import { structureMultipleChoice } from '../multiple-choice/fenwick-tree/structure';
import { operationsMultipleChoice } from '../multiple-choice/fenwick-tree/operations';
import { advancedMultipleChoice } from '../multiple-choice/fenwick-tree/advanced';
import { comparisonMultipleChoice } from '../multiple-choice/fenwick-tree/comparison';
import { interviewstrategyMultipleChoice } from '../multiple-choice/fenwick-tree/interview-strategy';

export const fenwickTreeModule: Module = {
  id: 'fenwick-tree',
  title: 'Fenwick Tree (Binary Indexed Tree)',
  description:
    'Master Fenwick Trees for efficient prefix sum queries and updates.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🎯',
  keyTakeaways: [
    'Fenwick Tree (BIT) provides O(log N) prefix sums and updates',
    'Uses bit manipulation: i & -i gets last set bit',
    'Update: i += i & -i (move to parent), Query: i -= i & -i (move back)',
    'Simpler than Segment Tree but only works for operations with inverse',
    '1-indexed: tree[0] unused, indices 1 to N',
    'Perfect for: range sums, counting inversions, cumulative frequency',
    'Cannot do: min/max queries (no inverse operation)',
    'Space: O(N), much simpler code than Segment Tree (~20 lines)',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...structureSection,
      quiz: structureQuiz,
      multipleChoice: structureMultipleChoice,
    },
    {
      ...operationsSection,
      quiz: operationsQuiz,
      multipleChoice: operationsMultipleChoice,
    },
    {
      ...advancedSection,
      quiz: advancedQuiz,
      multipleChoice: advancedMultipleChoice,
    },
    {
      ...comparisonSection,
      quiz: comparisonQuiz,
      multipleChoice: comparisonMultipleChoice,
    },
    {
      ...interviewstrategySection,
      quiz: interviewstrategyQuiz,
      multipleChoice: interviewstrategyMultipleChoice,
    },
  ],
};
