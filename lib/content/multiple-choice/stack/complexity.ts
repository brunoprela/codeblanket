/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the time complexity of push() and pop() operations on a stack?',
    options: ['O(log N)', 'O(1)', 'O(N)', 'O(N log N)'],
    correctAnswer: 1,
    explanation:
      'Push and pop operations on a stack are O(1) constant time because they only access the top element without traversing the rest of the stack.',
  },
  {
    id: 'mc2',
    question: 'How does a monotonic stack achieve O(N) time complexity?',
    options: [
      'By sorting the elements',
      'Each element is pushed and popped at most once',
      'By using binary search',
      'By processing elements backwards',
    ],
    correctAnswer: 1,
    explanation:
      'Each element is pushed exactly once and popped at most once during the entire algorithm, giving 2N total operations, which is O(N) amortized.',
  },
  {
    id: 'mc3',
    question:
      'What is the worst-case space complexity of a stack solution for valid parentheses?',
    options: ['O(1)', 'O(N)', 'O(log N)', 'O(N²)'],
    correctAnswer: 1,
    explanation:
      'Worst case occurs when all brackets are opening brackets (e.g., "(((("), requiring O(N) space to store them all in the stack.',
  },
  {
    id: 'mc4',
    question: 'Why is amortized analysis important for stack problems?',
    options: [
      'It makes the code faster',
      'Individual operations vary but average to O(1) across many operations',
      'It reduces space complexity',
      'It is not important',
    ],
    correctAnswer: 1,
    explanation:
      'Amortized analysis shows that even though some operations might seem expensive (like popping many elements), each element is processed a constant number of times total, averaging to O(1) per operation.',
  },
  {
    id: 'mc5',
    question: 'How do stacks transform O(N²) problems into O(N)?',
    options: [
      'By sorting the data first',
      'By remembering information to avoid repeated backward scans',
      'By using parallel processing',
      'They cannot do this',
    ],
    correctAnswer: 1,
    explanation:
      'Stacks maintain useful state information, eliminating the need for nested loops that repeatedly scan backwards. Each element is processed a constant number of times.',
  },
];
