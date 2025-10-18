/**
 * Multiple choice questions for Introduction to Queues section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does FIFO stand for?',
    options: [
      'First-In-First-Out',
      'Fast-In-Fast-Out',
      'Final-In-Final-Out',
      'First-Item-First-Operation',
    ],
    correctAnswer: 0,
    explanation:
      'FIFO stands for First-In-First-Out, meaning the first element added is the first one removed, like people in a line.',
  },
  {
    id: 'mc2',
    question: 'Where are elements added and removed in a queue?',
    options: [
      'Added and removed from the front',
      'Added and removed from the rear',
      'Added at rear, removed from front',
      'Added at front, removed from rear',
    ],
    correctAnswer: 2,
    explanation:
      'Elements are enqueued (added) at the rear and dequeued (removed) from the front, maintaining FIFO order.',
  },
  {
    id: 'mc3',
    question: 'Which real-world scenario best represents a queue?',
    options: [
      'Stack of plates',
      'Pile of books',
      'Line at a store checkout',
      'Undo button in editor',
    ],
    correctAnswer: 2,
    explanation:
      'A checkout line is a perfect queue analogy - first person in line is first served (FIFO). Stacks of plates/books use LIFO.',
  },
  {
    id: 'mc4',
    question: 'What are the two primary operations of a queue?',
    options: [
      'Push and Pop',
      'Enqueue and Dequeue',
      'Insert and Delete',
      'Add and Remove',
    ],
    correctAnswer: 1,
    explanation:
      'The two primary queue operations are enqueue (add to rear) and dequeue (remove from front).',
  },
  {
    id: 'mc5',
    question:
      'Which data structure is the opposite of a queue in terms of ordering?',
    options: ['Array', 'Linked List', 'Stack', 'Tree'],
    correctAnswer: 2,
    explanation:
      'A stack is the opposite of a queue: queue is FIFO (first in, first out) while stack is LIFO (last in, first out).',
  },
];
