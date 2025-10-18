/**
 * Depth-First Search (DFS) Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/dfs/introduction';
import { treedfsSection } from '../sections/dfs/tree-dfs';
import { graphdfsSection } from '../sections/dfs/graph-dfs';
import { iterativedfsSection } from '../sections/dfs/iterative-dfs';
import { complexitySection } from '../sections/dfs/complexity';
import { patternsSection } from '../sections/dfs/patterns';
import { interviewstrategySection } from '../sections/dfs/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/dfs/introduction';
import { treedfsQuiz } from '../quizzes/dfs/tree-dfs';
import { graphdfsQuiz } from '../quizzes/dfs/graph-dfs';
import { iterativedfsQuiz } from '../quizzes/dfs/iterative-dfs';
import { complexityQuiz } from '../quizzes/dfs/complexity';
import { patternsQuiz } from '../quizzes/dfs/patterns';
import { interviewstrategyQuiz } from '../quizzes/dfs/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/dfs/introduction';
import { treedfsMultipleChoice } from '../multiple-choice/dfs/tree-dfs';
import { graphdfsMultipleChoice } from '../multiple-choice/dfs/graph-dfs';
import { iterativedfsMultipleChoice } from '../multiple-choice/dfs/iterative-dfs';
import { complexityMultipleChoice } from '../multiple-choice/dfs/complexity';
import { patternsMultipleChoice } from '../multiple-choice/dfs/patterns';
import { interviewstrategyMultipleChoice } from '../multiple-choice/dfs/interview-strategy';

export const dfsModule: Module = {
  id: 'dfs',
  title: 'Depth-First Search (DFS)',
  description:
    'Master depth-first search for exploring trees and graphs by going as deep as possible before backtracking.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🌊',
  keyTakeaways: [
    'DFS explores as deep as possible before backtracking using recursion or a stack',
    'Tree DFS: Preorder (Root→L→R), Inorder (L→Root→R), Postorder (L→R→Root)',
    'Graph DFS requires visited set to avoid cycles, time is O(V+E)',
    'Recursive DFS uses O(H) space for call stack, where H is height',
    'Use DFS for: tree traversals, finding all paths, backtracking, cycle detection',
    'Top-down DFS passes info down, bottom-up DFS returns info up',
    'Iterative DFS with explicit stack avoids recursion depth limits',
    'DFS is natural for problems requiring exhaustive exploration or path tracking',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...treedfsSection,
      quiz: treedfsQuiz,
      multipleChoice: treedfsMultipleChoice,
    },
    {
      ...graphdfsSection,
      quiz: graphdfsQuiz,
      multipleChoice: graphdfsMultipleChoice,
    },
    {
      ...iterativedfsSection,
      quiz: iterativedfsQuiz,
      multipleChoice: iterativedfsMultipleChoice,
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
      ...interviewstrategySection,
      quiz: interviewstrategyQuiz,
      multipleChoice: interviewstrategyMultipleChoice,
    },
  ],
};
