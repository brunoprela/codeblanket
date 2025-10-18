/**
 * Advanced Graphs Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/advanced-graphs/introduction';
import { dijkstraSection } from '../sections/advanced-graphs/dijkstra';
import { bellmanfordSection } from '../sections/advanced-graphs/bellman-ford';
import { floydwarshallSection } from '../sections/advanced-graphs/floyd-warshall';
import { unionfindSection } from '../sections/advanced-graphs/union-find';
import { mstSection } from '../sections/advanced-graphs/mst';
import { comparisonSection } from '../sections/advanced-graphs/comparison';
import { interviewSection } from '../sections/advanced-graphs/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/advanced-graphs/introduction';
import { dijkstraQuiz } from '../quizzes/advanced-graphs/dijkstra';
import { bellmanfordQuiz } from '../quizzes/advanced-graphs/bellman-ford';
import { floydwarshallQuiz } from '../quizzes/advanced-graphs/floyd-warshall';
import { unionfindQuiz } from '../quizzes/advanced-graphs/union-find';
import { mstQuiz } from '../quizzes/advanced-graphs/mst';
import { comparisonQuiz } from '../quizzes/advanced-graphs/comparison';
import { interviewQuiz } from '../quizzes/advanced-graphs/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/advanced-graphs/introduction';
import { dijkstraMultipleChoice } from '../multiple-choice/advanced-graphs/dijkstra';
import { bellmanfordMultipleChoice } from '../multiple-choice/advanced-graphs/bellman-ford';
import { floydwarshallMultipleChoice } from '../multiple-choice/advanced-graphs/floyd-warshall';
import { unionfindMultipleChoice } from '../multiple-choice/advanced-graphs/union-find';
import { mstMultipleChoice } from '../multiple-choice/advanced-graphs/mst';
import { comparisonMultipleChoice } from '../multiple-choice/advanced-graphs/comparison';
import { interviewMultipleChoice } from '../multiple-choice/advanced-graphs/interview';

export const advancedGraphsModule: Module = {
  id: 'advanced-graphs',
  title: 'Advanced Graphs',
  description:
    'Master advanced graph algorithms including shortest paths, minimum spanning trees, and network flow.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🗺️',
  keyTakeaways: [
    "Dijkstra's: Fastest for single-source with non-negative weights - O((V+E)logV)",
    'Bellman-Ford: Handles negative weights and detects cycles - O(VE)',
    'Floyd-Warshall: All-pairs shortest path using DP - O(V³)',
    'Use BFS for unweighted graphs (simplest)',
    'Dijkstra uses min-heap and greedy approach (always extend shortest path)',
    'Bellman-Ford relaxes all edges V-1 times',
    'Floyd-Warshall tries each vertex as intermediate point',
    'Never use Dijkstra with negative weights - it will fail!',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...dijkstraSection,
      quiz: dijkstraQuiz,
      multipleChoice: dijkstraMultipleChoice,
    },
    {
      ...bellmanfordSection,
      quiz: bellmanfordQuiz,
      multipleChoice: bellmanfordMultipleChoice,
    },
    {
      ...floydwarshallSection,
      quiz: floydwarshallQuiz,
      multipleChoice: floydwarshallMultipleChoice,
    },
    {
      ...unionfindSection,
      quiz: unionfindQuiz,
      multipleChoice: unionfindMultipleChoice,
    },
    {
      ...mstSection,
      quiz: mstQuiz,
      multipleChoice: mstMultipleChoice,
    },
    {
      ...comparisonSection,
      quiz: comparisonQuiz,
      multipleChoice: comparisonMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
