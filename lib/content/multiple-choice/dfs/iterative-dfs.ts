/**
 * Multiple choice questions for Iterative DFS with Stack section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const iterativedfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why use iterative DFS instead of recursive?',
    options: [
      'Always faster',
      'Avoid stack overflow for deep graphs, more explicit control',
      'Random',
      'Simpler code',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative DFS with explicit stack avoids recursion depth limits (Python ~1000 frames). Useful for very deep graphs. Also gives more control (can pause/resume). Trade-off: less natural code.',
  },
  {
    id: 'mc2',
    question: 'How do you implement iterative DFS preorder?',
    options: [
      'Use queue',
      'Stack: push root, loop (pop, process, push right, push left)',
      'Impossible',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative DFS: explicit stack, push root. Loop: pop node, process (preorder), push right child then left (so left pops first for LIFO). Matches recursive preorder.',
  },
  {
    id: 'mc3',
    question: 'How does iterative DFS differ for graph vs tree?',
    options: [
      'No difference',
      "Graph needs visited set to avoid cycles; tree doesn't",
      'Cannot do graphs',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Tree: no cycles, no visited set needed. Graph: must track visited to avoid infinite loops from cycles. Same stack-based approach, just add visited check before pushing neighbors.',
  },
  {
    id: 'mc4',
    question: 'Can you do iterative inorder DFS?',
    options: [
      'No',
      'Yes: go left as far as possible, process, then go right (trickier than preorder)',
      'Only recursive',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Iterative inorder: push all left children to stack, pop & process top, move to right child, repeat. More complex than preorder because need to process node between left and right subtrees.',
  },
  {
    id: 'mc5',
    question: 'What is the space complexity of iterative vs recursive DFS?',
    options: [
      'Iterative uses less',
      'Both O(H) - explicit stack vs call stack, same space',
      'Recursive uses less',
      'Iterative O(1)',
    ],
    correctAnswer: 1,
    explanation:
      'Both use O(H) space where H is depth. Recursive uses call stack (implicit), iterative uses explicit stack. Same asymptotic space, but iterative may have higher constants (storing full nodes vs frames).',
  },
];
