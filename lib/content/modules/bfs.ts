/**
 * Breadth-First Search (BFS) Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/bfs/introduction';
import { treebfsSection } from '../sections/bfs/tree-bfs';
import { graphbfsSection } from '../sections/bfs/graph-bfs';
import { shortestpathSection } from '../sections/bfs/shortest-path';
import { complexitySection } from '../sections/bfs/complexity';
import { patternsSection } from '../sections/bfs/patterns';
import { interviewstrategySection } from '../sections/bfs/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/bfs/introduction';
import { treebfsQuiz } from '../quizzes/bfs/tree-bfs';
import { graphbfsQuiz } from '../quizzes/bfs/graph-bfs';
import { shortestpathQuiz } from '../quizzes/bfs/shortest-path';
import { complexityQuiz } from '../quizzes/bfs/complexity';
import { patternsQuiz } from '../quizzes/bfs/patterns';
import { interviewstrategyQuiz } from '../quizzes/bfs/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/bfs/introduction';
import { treebfsMultipleChoice } from '../multiple-choice/bfs/tree-bfs';
import { graphbfsMultipleChoice } from '../multiple-choice/bfs/graph-bfs';
import { shortestpathMultipleChoice } from '../multiple-choice/bfs/shortest-path';
import { complexityMultipleChoice } from '../multiple-choice/bfs/complexity';
import { patternsMultipleChoice } from '../multiple-choice/bfs/patterns';
import { interviewstrategyMultipleChoice } from '../multiple-choice/bfs/interview-strategy';

export const bfsModule: Module = {
  id: 'bfs',
  title: 'Breadth-First Search (BFS)',
  description:
    'Master breadth-first search for level-by-level traversal and finding shortest paths in unweighted graphs.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📊',
  keyTakeaways: [
    'BFS explores level by level using a queue (FIFO), visiting nearest nodes first',
    'BFS finds shortest path in unweighted graphs - first visit guarantees shortest',
    'Time: O(V+E) for graphs, O(N) for trees; Space: O(W) for queue width',
    'Level-by-level processing: capture queue size before inner loop',
    'Multi-source BFS starts from multiple nodes simultaneously',
    'Use BFS for: shortest path, level-order traversal, minimum steps problems',
    'Always use visited set to avoid cycles in graphs',
    'Bidirectional BFS can reduce search space from O(b^d) to O(b^(d/2))',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...treebfsSection,
      quiz: treebfsQuiz,
      multipleChoice: treebfsMultipleChoice,
    },
    {
      ...graphbfsSection,
      quiz: graphbfsQuiz,
      multipleChoice: graphbfsMultipleChoice,
    },
    {
      ...shortestpathSection,
      quiz: shortestpathQuiz,
      multipleChoice: shortestpathMultipleChoice,
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
