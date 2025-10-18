/**
 * Recursion Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/recursion/introduction';
import { anatomySection } from '../sections/recursion/anatomy';
import { patternsSection } from '../sections/recursion/patterns';
import { recursionvsiterationSection } from '../sections/recursion/recursion-vs-iteration';
import { memoizationSection } from '../sections/recursion/memoization';
import { debuggingrecursionSection } from '../sections/recursion/debugging-recursion';
import { interviewstrategySection } from '../sections/recursion/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/recursion/introduction';
import { anatomyQuiz } from '../quizzes/recursion/anatomy';
import { patternsQuiz } from '../quizzes/recursion/patterns';
import { recursionvsiterationQuiz } from '../quizzes/recursion/recursion-vs-iteration';
import { memoizationQuiz } from '../quizzes/recursion/memoization';
import { debuggingrecursionQuiz } from '../quizzes/recursion/debugging-recursion';
import { interviewstrategyQuiz } from '../quizzes/recursion/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/recursion/introduction';
import { anatomyMultipleChoice } from '../multiple-choice/recursion/anatomy';
import { patternsMultipleChoice } from '../multiple-choice/recursion/patterns';
import { recursionvsiterationMultipleChoice } from '../multiple-choice/recursion/recursion-vs-iteration';
import { memoizationMultipleChoice } from '../multiple-choice/recursion/memoization';
import { debuggingrecursionMultipleChoice } from '../multiple-choice/recursion/debugging-recursion';
import { interviewstrategyMultipleChoice } from '../multiple-choice/recursion/interview-strategy';

export const recursionModule: Module = {
  id: 'recursion',
  title: 'Recursion',
  description:
    'Master recursion from basics to advanced - the foundation for DFS, backtracking, and dynamic programming.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔄',
  keyTakeaways: [
    'Recursion solves problems by breaking them into smaller identical subproblems',
    'Every recursive function needs a base case to prevent infinite recursion',
    'Call stack grows with recursion depth: O(n) space for n recursive calls',
    'Memoization caches results to avoid redundant calculations',
    'Use @lru_cache decorator in Python for automatic memoization',
    'Tail recursion can be optimized by some languages (but not Python)',
    'Common patterns: divide-and-conquer, backtracking, tree traversal',
    'Watch for exponential time complexity without memoization (e.g., naive Fibonacci)',
    'Consider iteration if stack overflow is a concern or language lacks tail call optimization',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...anatomySection,
      quiz: anatomyQuiz,
      multipleChoice: anatomyMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
    },
    {
      ...recursionvsiterationSection,
      quiz: recursionvsiterationQuiz,
      multipleChoice: recursionvsiterationMultipleChoice,
    },
    {
      ...memoizationSection,
      quiz: memoizationQuiz,
      multipleChoice: memoizationMultipleChoice,
    },
    {
      ...debuggingrecursionSection,
      quiz: debuggingrecursionQuiz,
      multipleChoice: debuggingrecursionMultipleChoice,
    },
    {
      ...interviewstrategySection,
      quiz: interviewstrategyQuiz,
      multipleChoice: interviewstrategyMultipleChoice,
    },
  ],
};
