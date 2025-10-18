/**
 * Multiple choice questions for Fenwick Tree vs Segment Tree section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const comparisonMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'When should you use Fenwick Tree over Segment Tree?',
    options: [
      'Always',
      'Simpler problem (sum/XOR), want shorter code, operation has inverse',
      'Never',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      "Use Fenwick when: 1) Operation is invertible (sum, XOR), 2) Don't need min/max/GCD, 3) Want simpler code (~20 lines vs 50+), 4) Interview time pressure. Same O(log N) complexity.",
  },
  {
    id: 'mc2',
    question: 'When must you use Segment Tree instead of Fenwick Tree?',
    options: [
      'Random',
      'Need min/max/GCD (no inverse) or lazy propagation for range updates',
      'Always',
      'Never',
    ],
    correctAnswer: 1,
    explanation:
      'Use Segment Tree when: 1) Operation has no inverse (min, max, GCD), 2) Need lazy propagation for efficient range updates, 3) Complex custom operations. Fenwick is limited to invertible operations.',
  },
  {
    id: 'mc3',
    question: 'What are the code complexity differences?',
    options: [
      'Same',
      'Fenwick: ~20 lines simple, Segment: ~50+ lines complex',
      'Fenwick longer',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree: 2 simple functions, ~20 lines, straightforward bit manipulation. Segment Tree: recursive build/query/update, lazy propagation, ~50+ lines, more complex. Big difference in interview settings.',
  },
  {
    id: 'mc4',
    question: 'What memory differences exist?',
    options: [
      'Same',
      'Fenwick: N+1 array, Segment: 4N array - Fenwick uses 25% space',
      'Segment uses less',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fenwick Tree uses N+1 space (1-indexed array). Segment Tree uses 4N for safety. Fenwick is 4x more space-efficient, though both are O(N).',
  },
  {
    id: 'mc5',
    question: 'Performance in practice?',
    options: [
      'Segment always faster',
      'Similar O(log N), but Fenwick often faster due to simpler operations and better constants',
      'Fenwick always slower',
      'No difference',
    ],
    correctAnswer: 1,
    explanation:
      'Both O(log N), but Fenwick often faster in practice: simpler operations (just addition + bit manipulation), better cache locality (smaller structure), lower constant factors.',
  },
];
