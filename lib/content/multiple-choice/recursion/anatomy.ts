/**
 * Multiple choice questions for Anatomy of a Recursive Function section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const anatomyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of binary search recursion over linear search?',
    options: [
      'It uses less memory',
      'It reduces time complexity from O(n) to O(log n)',
      'It works on unsorted arrays',
      'It is easier to implement',
    ],
    correctAnswer: 1,
    explanation:
      "Binary search recursively divides the search space in half each time, achieving O(log n) time complexity compared to linear search's O(n).",
  },
  {
    id: 'mc2',
    question:
      'In recursive list operations, what is typically the base case for an empty list?',
    options: [
      'Return the first element',
      'Return None or an appropriate empty value',
      'Raise an exception',
      'Call the function again',
    ],
    correctAnswer: 1,
    explanation:
      'For empty lists, the base case typically returns an appropriate empty value (e.g., 0 for sum, [] for filtering) or None, depending on the operation.',
  },
  {
    id: 'mc3',
    question: 'What makes tree traversal naturally suited for recursion?',
    options: [
      'Trees are always balanced',
      'Trees have a recursive structure (node with subtrees)',
      'Trees are stored in arrays',
      'Recursion is faster for trees',
    ],
    correctAnswer: 1,
    explanation:
      'Trees are recursively defined: each node has left and right subtrees, which are themselves trees. This structure maps naturally to recursive solutions.',
  },
  {
    id: 'mc4',
    question: 'When should you prefer iteration over recursion?',
    options: [
      'When the problem has a natural recursive structure',
      'When recursion depth might cause stack overflow',
      'When working with trees or graphs',
      'Never, recursion is always better',
    ],
    correctAnswer: 1,
    explanation:
      'Use iteration when recursion might be too deep (risk of stack overflow) or when iteration is clearer. Recursion is better for naturally recursive structures.',
  },
  {
    id: 'mc5',
    question: 'What is the "leap of faith" in recursive thinking?',
    options: [
      'Hoping the code will work',
      'Trusting that the recursive call solves the smaller problem correctly',
      'Skipping the base case',
      'Not testing edge cases',
    ],
    correctAnswer: 1,
    explanation:
      'The "leap of faith" means assuming the recursive call works correctly for smaller inputs, allowing you to focus on: (1) the base case, and (2) combining the recursive result to solve the current problem.',
  },
];
