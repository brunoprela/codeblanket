/**
 * Sorting Algorithms Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/sorting/introduction';
import { comparisonsortsSection } from '../sections/sorting/comparison-sorts';
import { noncomparisonsortsSection } from '../sections/sorting/non-comparison-sorts';
import { practicalconsiderationsSection } from '../sections/sorting/practical-considerations';
import { interviewproblemsSection } from '../sections/sorting/interview-problems';
import { comparisonguideSection } from '../sections/sorting/comparison-guide';

// Import quizzes
import { introductionQuiz } from '../quizzes/sorting/introduction';
import { comparisonsortsQuiz } from '../quizzes/sorting/comparison-sorts';
import { noncomparisonsortsQuiz } from '../quizzes/sorting/non-comparison-sorts';
import { practicalconsiderationsQuiz } from '../quizzes/sorting/practical-considerations';
import { interviewproblemsQuiz } from '../quizzes/sorting/interview-problems';
import { comparisonguideQuiz } from '../quizzes/sorting/comparison-guide';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/sorting/introduction';
import { comparisonsortsMultipleChoice } from '../multiple-choice/sorting/comparison-sorts';
import { noncomparisonsortsMultipleChoice } from '../multiple-choice/sorting/non-comparison-sorts';
import { practicalconsiderationsMultipleChoice } from '../multiple-choice/sorting/practical-considerations';
import { interviewproblemsMultipleChoice } from '../multiple-choice/sorting/interview-problems';
import { comparisonguideMultipleChoice } from '../multiple-choice/sorting/comparison-guide';

export const sortingModule: Module = {
  id: 'sorting',
  title: 'Sorting Algorithms',
  description:
    'Master the fundamental sorting algorithms and understand their time complexity, space usage, and when to use each.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔄',
  keyTakeaways: [
    'Comparison sorts are at least O(n log n) average case - fundamental lower bound',
    'Simple sorts (bubble, selection, insertion) are O(n²) but useful for small data',
    'Efficient sorts (merge, quick, heap) are O(n log n) with different tradeoffs',
    'Non-comparison sorts (counting, radix, bucket) can achieve O(n) for specific data types',
    'Stability matters when sorting by multiple criteria or preserving order',
    'Real-world implementations use hybrid algorithms like Timsort and Introsort',
    'Quick sort is fastest in practice but has O(n²) worst case; merge sort guarantees O(n log n)',
    'For top-k problems, use heaps (O(n log k)) instead of full sorting (O(n log n))',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...comparisonsortsSection,
      quiz: comparisonsortsQuiz,
      multipleChoice: comparisonsortsMultipleChoice,
    },
    {
      ...noncomparisonsortsSection,
      quiz: noncomparisonsortsQuiz,
      multipleChoice: noncomparisonsortsMultipleChoice,
    },
    {
      ...practicalconsiderationsSection,
      quiz: practicalconsiderationsQuiz,
      multipleChoice: practicalconsiderationsMultipleChoice,
    },
    {
      ...interviewproblemsSection,
      quiz: interviewproblemsQuiz,
      multipleChoice: interviewproblemsMultipleChoice,
    },
    {
      ...comparisonguideSection,
      quiz: comparisonguideQuiz,
      multipleChoice: comparisonguideMultipleChoice,
    },
  ],
};
