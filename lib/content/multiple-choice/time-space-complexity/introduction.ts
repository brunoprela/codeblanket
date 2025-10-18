/**
 * Multiple choice questions for What is Complexity Analysis? section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does Big O notation describe?',
    options: [
      'The exact number of operations an algorithm performs',
      "How an algorithm's resource requirements grow with input size",
      'The amount of memory an algorithm uses',
      'The programming language used to implement an algorithm',
    ],
    correctAnswer: 1,
    explanation:
      "Big O notation describes how an algorithm's time or space requirements grow as the input size increases. It focuses on growth rate, not exact counts or specific implementations.",
  },
  {
    id: 'mc2',
    question: 'Which complexity is best (fastest) for large inputs?',
    options: ['O(n²)', 'O(n log n)', 'O(n)', 'O(log n)'],
    correctAnswer: 3,
    explanation:
      'O(log n) is the fastest among these options. Logarithmic time complexity grows very slowly as input size increases, making it ideal for large datasets.',
  },
  {
    id: 'mc3',
    question: 'Why do we drop constants in Big O notation?',
    options: [
      'To make the math easier',
      'Because constants are always small',
      'To focus on the growth rate as input size approaches infinity',
      'Because different computers run at different speeds',
    ],
    correctAnswer: 2,
    explanation:
      'We drop constants because Big O notation focuses on how the algorithm scales with input size. As n approaches infinity, constant factors become less significant compared to the growth rate itself.',
  },
  {
    id: 'mc4',
    question:
      'If an algorithm has time complexity O(n) and space complexity O(1), what does this mean?',
    options: [
      'It takes constant time and linear space',
      'It takes linear time and constant space',
      'It takes constant time and constant space',
      'It takes linear time and linear space',
    ],
    correctAnswer: 1,
    explanation:
      "O(n) time means the algorithm's execution time grows linearly with input size. O(1) space means it uses a constant amount of memory regardless of input size.",
  },
  {
    id: 'mc5',
    question:
      'What is the time complexity of accessing an element in an array by index (e.g., arr[5])?',
    options: ['O(1)', 'O(log n)', 'O(n)', 'O(n²)'],
    correctAnswer: 0,
    explanation:
      'Array access by index is O(1) - constant time. The computer can calculate the memory address directly using the base address plus the index, regardless of array size.',
  },
];
