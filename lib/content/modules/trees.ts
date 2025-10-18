/**
 * Trees Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/trees/introduction';
import { traversalsSection } from '../sections/trees/traversals';
import { patternsSection } from '../sections/trees/patterns';
import { complexitySection } from '../sections/trees/complexity';
import { templatesSection } from '../sections/trees/templates';
import { lowestcommonancestorSection } from '../sections/trees/lowest-common-ancestor';
import { interviewSection } from '../sections/trees/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/trees/introduction';
import { traversalsQuiz } from '../quizzes/trees/traversals';
import { patternsQuiz } from '../quizzes/trees/patterns';
import { complexityQuiz } from '../quizzes/trees/complexity';
import { templatesQuiz } from '../quizzes/trees/templates';
import { lowestcommonancestorQuiz } from '../quizzes/trees/lowest-common-ancestor';
import { interviewQuiz } from '../quizzes/trees/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/trees/introduction';
import { traversalsMultipleChoice } from '../multiple-choice/trees/traversals';
import { patternsMultipleChoice } from '../multiple-choice/trees/patterns';
import { complexityMultipleChoice } from '../multiple-choice/trees/complexity';
import { templatesMultipleChoice } from '../multiple-choice/trees/templates';
import { lowestcommonancestorMultipleChoice } from '../multiple-choice/trees/lowest-common-ancestor';
import { interviewMultipleChoice } from '../multiple-choice/trees/interview';

export const treesModule: Module = {
  id: 'trees',
  title: 'Trees',
  description:
    'Master tree structures, traversals, and recursive problem-solving for hierarchical data.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🌳',
  keyTakeaways: [
    'Trees are hierarchical: each node has 0+ children, with one root and no cycles',
    'Binary trees: each node has ≤ 2 children (left, right)',
    'BST property: left < root < right for all nodes enables O(log N) operations',
    'DFS traversals: preorder (root first), inorder (root middle, sorted for BST), postorder (root last)',
    'BFS/level-order: process nodes level by level using a queue',
    'Recursive solutions: define base case (null), recursively solve subtrees, combine results',
    'Time: O(N) for traversals; Space: O(H) for recursion where H = log N (balanced) to N (skewed)',
    'Use DFS for path problems, BST for search/insert, BFS for level-specific operations',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...traversalsSection,
      quiz: traversalsQuiz,
      multipleChoice: traversalsMultipleChoice,
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
      ...lowestcommonancestorSection,
      quiz: lowestcommonancestorQuiz,
      multipleChoice: lowestcommonancestorMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
