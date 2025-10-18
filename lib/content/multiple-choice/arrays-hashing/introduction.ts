/**
 * Multiple choice questions for Arrays & Hash Tables: The Foundation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary advantage of hash tables over arrays?',
    options: [
      'Hash tables use less memory',
      'Hash tables maintain sorted order',
      'Hash tables provide O(1) average-case lookup',
      'Hash tables can only store unique values',
    ],
    correctAnswer: 2,
    explanation:
      'Hash tables provide O(1) average-case lookup, insert, and delete operations, compared to O(n) for unsorted arrays. This makes them ideal for problems requiring frequent lookups or existence checks.',
  },
  {
    id: 'mc2',
    question:
      'In the two sum problem, why does using a hash table reduce time complexity from O(n²) to O(n)?',
    options: [
      'Hash tables sort the data automatically',
      'Hash tables eliminate the need for a nested loop by providing O(1) lookups',
      'Hash tables reduce the input size',
      'Hash tables only store unique elements',
    ],
    correctAnswer: 1,
    explanation:
      'The brute force approach uses nested loops to check all pairs (O(n²)). With a hash table, we can check if the complement exists in O(1) time, allowing us to solve the problem in a single pass through the array.',
  },
  {
    id: 'mc3',
    question:
      'Which data structure should you use for counting element frequencies?',
    options: ['List', 'Set', 'Counter or Dictionary', 'Tuple'],
    correctAnswer: 2,
    explanation:
      'Counter (from collections) or a regular dictionary is perfect for counting frequencies. Sets only track existence, lists require O(n) search, and tuples are immutable.',
  },
  {
    id: 'mc4',
    question:
      'What percentage of interview problems use arrays or hash tables?',
    options: ['10-20%', '20-30%', '30-40%', '50-60%'],
    correctAnswer: 2,
    explanation:
      'Arrays and hash tables appear in approximately 30-40% of all coding interview problems, making them the most fundamental data structures to master.',
  },
  {
    id: 'mc5',
    question:
      'What is the space complexity when using a hash table to solve the two sum problem?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 2,
    explanation:
      'We store up to n elements in the hash table in the worst case (when no pair is found or the pair is at the end), resulting in O(n) space complexity.',
  },
];
