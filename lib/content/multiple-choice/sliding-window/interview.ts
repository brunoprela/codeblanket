/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What keywords in a problem statement strongly suggest using sliding window?',
    options: [
      'Recursive, tree, graph',
      'Contiguous, subarray, substring, consecutive',
      'Binary, sorted, search',
      'Hash, frequency, count',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "contiguous", "subarray", "substring", and "consecutive" strongly indicate sliding window is applicable, as they describe sequential elements that form a window.',
  },
  {
    id: 'mc2',
    question:
      'When explaining a sliding window solution, what should you communicate first?',
    options: [
      'The code implementation',
      'The complexity analysis',
      'Problem recognition and chosen window type (fixed vs variable)',
      'Test cases',
    ],
    correctAnswer: 2,
    explanation:
      "Start by recognizing the problem pattern and explaining which window type you'll use and why. This shows your thought process and sets up the solution clearly.",
  },
  {
    id: 'mc3',
    question: 'What is a good practice strategy for mastering sliding window?',
    options: [
      'Only practice hard problems',
      'Start with fixed-size, then variable maximum, then variable minimum, then advanced',
      'Practice randomly',
      'Memorize all solutions',
    ],
    correctAnswer: 1,
    explanation:
      'Progress from simpler fixed-size windows to variable-size (maximum first, then minimum), and finally advanced techniques. This builds understanding incrementally.',
  },
  {
    id: 'mc4',
    question:
      'How long should a medium sliding window problem take in an interview?',
    options: [
      '5-10 minutes',
      '15-25 minutes',
      '30-40 minutes',
      '45-60 minutes',
    ],
    correctAnswer: 1,
    explanation:
      'Medium sliding window problems typically take 15-25 minutes including explanation, coding, and testing. This accounts for clear communication and verification.',
  },
  {
    id: 'mc5',
    question:
      'What is the key insight that makes sliding window O(n) instead of O(n²)?',
    options: [
      'Using better data structures',
      'Each element enters and leaves the window at most once',
      'Parallel processing',
      'Using recursion',
    ],
    correctAnswer: 1,
    explanation:
      'The key insight is that each element is processed at most twice: once when the right pointer includes it, once when the left pointer removes it. This gives O(n) total operations, not O(n²).',
  },
];
