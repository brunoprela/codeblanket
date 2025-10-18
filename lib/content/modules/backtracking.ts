/**
 * Backtracking Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/backtracking/introduction';
import { patternsSection } from '../sections/backtracking/patterns';
import { complexitySection } from '../sections/backtracking/complexity';
import { templatesSection } from '../sections/backtracking/templates';
import { interviewSection } from '../sections/backtracking/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/backtracking/introduction';
import { patternsQuiz } from '../quizzes/backtracking/patterns';
import { complexityQuiz } from '../quizzes/backtracking/complexity';
import { templatesQuiz } from '../quizzes/backtracking/templates';
import { interviewQuiz } from '../quizzes/backtracking/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/backtracking/introduction';
import { patternsMultipleChoice } from '../multiple-choice/backtracking/patterns';
import { complexityMultipleChoice } from '../multiple-choice/backtracking/complexity';
import { templatesMultipleChoice } from '../multiple-choice/backtracking/templates';
import { interviewMultipleChoice } from '../multiple-choice/backtracking/interview';

export const backtrackingModule: Module = {
  id: 'backtracking',
  title: 'Backtracking',
  description:
    'Master backtracking for exploring all possible solutions through exhaustive search with pruning.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔙',
  keyTakeaways: [
    'Backtracking explores all possibilities by making choices, exploring, and undoing (backtracking)',
    'Three steps: (1) make choice, (2) recurse with choice, (3) undo choice (backtrack)',
    'Subsets: include/exclude decision at each index, use start parameter to avoid duplicates',
    'Permutations: try all unused elements at each position, track with visited set or remaining list',
    'Combinations: like subsets but with fixed size K, stop when path reaches size K',
    'Constraint satisfaction: check validity before recursing (early pruning)',
    'Always copy path when adding to results (path[:] or path.copy())',
    'Time complexity typically O(2^N) for subsets, O(N!) for permutations, O(b^d) generally',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
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
