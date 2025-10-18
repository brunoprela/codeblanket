/**
 * Multiple choice questions for Common Pitfalls section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpitfallsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the most common mistake when working with linked lists?',
    options: [
      'Using too much memory',
      'Not checking for null/None before accessing node.next',
      'Making lists too long',
      'Using wrong variable names',
    ],
    correctAnswer: 1,
    explanation:
      'Forgetting null checks causes null pointer dereference errors. Always check "if curr and curr.next" before accessing curr.next.next to avoid crashes on empty lists or at list boundaries.',
  },
  {
    id: 'mc2',
    question:
      'What happens if you reverse a pointer before saving the next reference?',
    options: [
      'Nothing, it works fine',
      'You lose access to the rest of the list',
      'The list gets sorted',
      'Memory leak occurs',
    ],
    correctAnswer: 1,
    explanation:
      'If you do curr.next = prev before saving the original curr.next, you lose the reference to the rest of the list. Always save next with next_temp = curr.next first.',
  },
  {
    id: 'mc3',
    question:
      'Why should you use a dummy node when removing nodes from a linked list?',
    options: [
      'It makes the code slower',
      'It eliminates special case handling for removing the head node',
      'It is required by Python',
      'It reduces memory usage',
    ],
    correctAnswer: 1,
    explanation:
      'A dummy node eliminates the need for special case logic when removing the head node. With a dummy, all nodes (including the head at dummy.next) can be handled uniformly.',
  },
  {
    id: 'mc4',
    question:
      'What is the off-by-one error in the runner technique for finding kth from end?',
    options: [
      'Moving first pointer k-1 steps instead of k',
      'Moving first pointer k steps instead of k+1 or k-1',
      'Not moving any pointer',
      'Moving both pointers',
    ],
    correctAnswer: 0,
    explanation:
      'The first pointer should move exactly k steps ahead. Common mistakes include k-1 (one too few) or k+1 (one too many). Test with a small example: list [1,2,3,4,5], k=2 should return [4].',
  },
  {
    id: 'mc5',
    question: 'Why might modifying a linked list in-place be problematic?',
    options: [
      'It is always wrong',
      'The caller might need the original data or other parts of code might hold references',
      'It uses more memory',
      'It is slower',
    ],
    correctAnswer: 1,
    explanation:
      'In-place modification destroys the original data, which is problematic if the caller needs it or if other code holds references to nodes. Always clarify whether in-place modification is acceptable.',
  },
];
