/**
 * Multiple choice questions for Code Templates section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const templatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the three parts of the basic recursive DFS template?',
    options: [
      'Start, middle, end',
      'Null check (base case), recursive calls on children, combine results',
      'Initialize, process, return',
      'Read, compute, write',
    ],
    correctAnswer: 1,
    explanation:
      'The DFS template has three parts: 1) Null check for base case, 2) Recursive calls on children to divide the problem, 3) Combine results from children with current node to conquer.',
  },
  {
    id: 'mc2',
    question: 'In the BFS template, why do we track level size separately?',
    options: [
      'To make it faster',
      'To separate levels within the queue since it mixes current level with children',
      'To save memory',
      'It is not necessary',
    ],
    correctAnswer: 1,
    explanation:
      'Level size lets us process exactly one level at a time. Without it, we cannot tell when one level ends since the queue mixes current level nodes with their children added during processing.',
  },
  {
    id: 'mc3',
    question: 'When should you use a top-down recursive pattern?',
    options: [
      'Always',
      'When current node needs context from ancestors (pass down as parameters)',
      'When it is faster',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Use top-down when nodes need context from ancestors, like passing bounds for BST validation or accumulated sum for path problems. Information flows down as parameters.',
  },
  {
    id: 'mc4',
    question: 'When should you use a bottom-up recursive pattern?',
    options: [
      'When it uses less memory',
      'When parent needs results computed by children (return up)',
      'Always for trees',
      'Only for balanced trees',
    ],
    correctAnswer: 1,
    explanation:
      'Use bottom-up when parent needs results from children, like computing tree height or checking if balanced. Children compute and return values that parent uses.',
  },
  {
    id: 'mc5',
    question:
      'What is the key advantage of iterative DFS/BFS templates over recursive?',
    options: [
      'They are always faster',
      'Explicit stack/queue control and avoid stack overflow for deep trees',
      'They use less code',
      'They are easier to understand',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative templates give explicit control over the stack/queue and avoid stack overflow issues with very deep recursion. They are especially useful for very deep or unbalanced trees.',
  },
];
