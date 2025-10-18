/**
 * Multiple choice questions for Rate Limiting & Counters section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const ratelimitingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main disadvantage of Fixed Window rate limiting?',
    options: [
      'It is too slow',
      'It allows burst traffic at window boundaries',
      'It uses too much memory',
      'It cannot be implemented',
    ],
    correctAnswer: 1,
    explanation:
      'Fixed Window allows burst at boundaries. Users can make max requests at end of one window and max again at start of next window, potentially doubling the rate for a brief period. This is the classic boundary spike problem.',
  },
  {
    id: 'mc2',
    question:
      'Why do we use a deque for Hit Counter instead of a regular list?',
    options: [
      'Deques are faster for all operations',
      'Deques allow O(1) removal from front (for old timestamps)',
      'Deques use less memory',
      'Deques automatically sort data',
    ],
    correctAnswer: 1,
    explanation:
      'Deque (double-ended queue) allows O(1) popleft() to remove old timestamps from front. Regular list would need list.pop(0) which is O(N) because it shifts all elements. Since we frequently remove old timestamps, O(1) removal is critical.',
  },
  {
    id: 'mc3',
    question: 'In Token Bucket, what happens when tokens reach capacity?',
    options: [
      'The system crashes',
      'Tokens start decreasing',
      'Tokens stop accumulating (cap at capacity)',
      'Requests are denied',
    ],
    correctAnswer: 2,
    explanation:
      "When tokens reach capacity, they stop accumulating - there's a maximum burst allowed. This prevents unlimited token accumulation if API is unused for days. Example: capacity=100 means max burst of 100, even if unused for a year.",
  },
  {
    id: 'mc4',
    question:
      'What is the time complexity of getHits() in the bucket-based Hit Counter?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 0,
    explanation:
      'Bucket-based getHits() is O(1) because it scans exactly 300 buckets (fixed), regardless of how many hits occurred. O(300) = O(1) for constant-size window. This is an advantage over deque approach which is O(hits_in_window).',
  },
  {
    id: 'mc5',
    question: 'Why is Sliding Window Counter better than Fixed Window?',
    options: [
      'It uses less memory',
      'It prevents boundary spikes while using O(1) memory',
      'It is simpler to implement',
      'It is always more accurate',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding Window Counter (weighted approach) prevents boundary spikes like Sliding Log, but uses only O(1) memory like Fixed Window. It weighs previous and current window counts based on position in current window, giving good approximation with minimal memory.',
  },
];
