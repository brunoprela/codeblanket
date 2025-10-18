/**
 * Multiple choice questions for What is Binary Search? section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the absolute requirement for using binary search?',
    options: [
      'The array must be large',
      'The array must be sorted',
      'The array must have unique elements',
      'The array must be in ascending order only',
    ],
    correctAnswer: 1,
    explanation:
      'Binary search requires the array to be sorted (either ascending or descending) so that comparisons with the middle element can reliably determine which half to search.',
  },
  {
    id: 'mc2',
    question: 'Why is it called "binary" search?',
    options: [
      'It works with binary numbers',
      'At each step, it makes a binary (two-way) decision',
      'It uses binary trees',
      'It divides by 2',
    ],
    correctAnswer: 1,
    explanation:
      'The name comes from the binary (two-way) decision made at each step: is the target in the left half or the right half? This binary choice creates a binary decision tree.',
  },
  {
    id: 'mc3',
    question: 'Can you efficiently use binary search on a sorted linked list?',
    options: [
      'Yes, as long as it is sorted',
      'No, because finding the middle element takes O(N) time',
      'Yes, but only for small lists',
      'No, because linked lists cannot be sorted',
    ],
    correctAnswer: 1,
    explanation:
      'Binary search requires O(1) random access to find the middle element. In a linked list, finding the middle takes O(N) time, eliminating the efficiency advantage.',
  },
  {
    id: 'mc4',
    question: 'What analogy best describes binary search?',
    options: [
      'Reading a book from start to finish',
      'Finding a word in a dictionary by repeatedly opening to the middle',
      'Sorting cards',
      'Counting items one by one',
    ],
    correctAnswer: 1,
    explanation:
      'Finding a word in a dictionary by opening to the middle and deciding which half to search next perfectly illustrates the binary search process.',
  },
  {
    id: 'mc5',
    question: 'What happens if you try binary search on an unsorted array?',
    options: [
      'It works but slower',
      'The algorithm breaks down because comparisons cannot reliably determine direction',
      'It automatically sorts the array first',
      'It still finds the element eventually',
    ],
    correctAnswer: 1,
    explanation:
      'Without sorting, comparing the target with the middle element gives no reliable information about which half contains the target, breaking the core logic of binary search.',
  },
];
