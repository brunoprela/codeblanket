/**
 * Graphs Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/graphs/introduction';
import { traversalsSection } from '../sections/graphs/traversals';
import { patternsSection } from '../sections/graphs/patterns';
import { complexitySection } from '../sections/graphs/complexity';
import { templatesSection } from '../sections/graphs/templates';
import { interviewSection } from '../sections/graphs/interview';
import { algorithmselectionSection } from '../sections/graphs/algorithm-selection';

// Import quizzes
import { introductionQuiz } from '../quizzes/graphs/introduction';
import { traversalsQuiz } from '../quizzes/graphs/traversals';
import { patternsQuiz } from '../quizzes/graphs/patterns';
import { complexityQuiz } from '../quizzes/graphs/complexity';
import { templatesQuiz } from '../quizzes/graphs/templates';
import { interviewQuiz } from '../quizzes/graphs/interview';
import { algorithmselectionQuiz } from '../quizzes/graphs/algorithm-selection';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/graphs/introduction';
import { traversalsMultipleChoice } from '../multiple-choice/graphs/traversals';
import { patternsMultipleChoice } from '../multiple-choice/graphs/patterns';
import { complexityMultipleChoice } from '../multiple-choice/graphs/complexity';
import { templatesMultipleChoice } from '../multiple-choice/graphs/templates';
import { interviewMultipleChoice } from '../multiple-choice/graphs/interview';
import { algorithmselectionMultipleChoice } from '../multiple-choice/graphs/algorithm-selection';

export const graphsModule: Module = {
  id: 'graphs',
  title: 'Graphs',
  description:
    'Master graph traversal, pathfinding, and connectivity problems for complex network structures.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🕸️',
  keyTakeaways: [
    'Graphs consist of vertices (nodes) connected by edges; can be directed/undirected, weighted/unweighted',
    'BFS explores level-by-level using queue; finds shortest path in unweighted graphs',
    'DFS explores deeply using stack/recursion; better for cycle detection and memory efficiency',
    'Adjacency list (dict of lists) is most common representation: O(V + E) space',
    'Most graph algorithms are O(V + E) time - linear in graph size',
    'Connected components: run DFS/BFS from each unvisited node',
    "Topological sort: order nodes in DAG using Kahn's algorithm (BFS with in-degrees)",
    'Union-Find provides near-constant time connectivity queries with path compression',
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
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
    {
      ...algorithmselectionSection,
      quiz: algorithmselectionQuiz,
      multipleChoice: algorithmselectionMultipleChoice,
    },
  ],
};
