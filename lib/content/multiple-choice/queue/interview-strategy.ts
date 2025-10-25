/**
 * Multiple choice questions for Queue Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewstrategyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which keyword in a problem description most strongly indicates using BFS with a queue?',
    options: [
      'Find all paths',
      'Shortest path in unweighted graph',
      'Detect cycles',
      'Generate permutations',
    ],
    correctAnswer: 1,
    explanation:
      'BFS with a queue is optimal for finding shortest paths in unweighted graphs because it explores nodes level by level, guaranteeing the first time you reach a node is via the shortest path.',
  },
  {
    id: 'mc2',
    question:
      'In the standard BFS template, when should you mark a node as visited?',
    options: [
      'After dequeuing it',
      'When adding it to the queue',
      'After processing all its neighbors',
      'Before the while loop starts',
    ],
    correctAnswer: 1,
    explanation:
      'Mark nodes as visited when adding them to the queue, not when dequeuing. This prevents adding the same node multiple times to the queue before it is processed.',
  },
  {
    id: 'mc3',
    question:
      'What is the purpose of capturing len (queue) in level-order traversal?',
    options: [
      'To check if the queue is empty',
      'To know how many nodes are in the current level',
      'To calculate total nodes in tree',
      'To prevent infinite loops',
    ],
    correctAnswer: 1,
    explanation:
      'Capturing len (queue) at the start tells us exactly how many nodes are in the current level, allowing us to process them separately from the next level.',
  },
  {
    id: 'mc4',
    question:
      'Which queue problem pattern is typically categorized as Hard difficulty?',
    options: [
      'Binary Tree Level Order Traversal',
      'Implement Queue using Stacks',
      'Sliding Window Median',
      'Number of Recent Calls',
    ],
    correctAnswer: 2,
    explanation:
      'Sliding Window Median requires maintaining two heaps (or balanced data structure) along with a sliding window, making it a Hard problem. Simple level-order and queue implementations are Easy/Medium.',
  },
  {
    id: 'mc5',
    question:
      'What is the most important thing to verify before saying your solution is complete in a queue interview?',
    options: [
      'The code compiles',
      'You used the right imports',
      'You analyzed time/space complexity correctly',
      'You added comments',
    ],
    correctAnswer: 2,
    explanation:
      'Always analyze and verify your time and space complexity. Interviewers want to see you understand not just that it works, but why it works and how efficient it is. This demonstrates algorithmic thinking.',
  },
];
