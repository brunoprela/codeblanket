/**
 * Multiple choice questions for Introduction to Math & Geometry section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What types of problems require math/geometry?',
    options: [
      'Only theoretical',
      'Number theory, primes, GCD, combinatorics, coordinates, angles, areas',
      'Random',
      'Never used',
    ],
    correctAnswer: 1,
    explanation:
      'Math/geometry in interviews: prime factorization, GCD/LCM, modular arithmetic, combinations/permutations, point distance, line intersection, polygon area. Common in competitive programming.',
  },
  {
    id: 'mc2',
    question: 'When should you recognize a math problem vs algorithmic?',
    options: [
      'Random',
      'Keywords: divisible, prime, factorial, angle, distance, area suggest math formulas over data structures',
      'Always algorithm',
      'No difference',
    ],
    correctAnswer: 1,
    explanation:
      'Math problem signals: "divisible by", "prime numbers", "GCD/LCM", "factorial", "angle", "distance", "area", "modulo". These suggest mathematical formulas/properties rather than complex data structures.',
  },
  {
    id: 'mc3',
    question:
      'What is the difference between coordinate and computational geometry?',
    options: [
      'Same',
      'Coordinate: points, distances, lines. Computational: algorithms for geometric structures (convex hull, closest pair)',
      'Random',
      'No geometry',
    ],
    correctAnswer: 1,
    explanation:
      'Coordinate geometry: basic operations with points/lines (distance, slope, intersection). Computational geometry: algorithmic problems (convex hull, line sweep, closest pair). Latter more advanced.',
  },
  {
    id: 'mc4',
    question: 'Why is modular arithmetic important in coding?',
    options: [
      'Random',
      'Prevents overflow, used in cryptography, handles large numbers (return answer mod 10^9+7)',
      'Only theoretical',
      'Not important',
    ],
    correctAnswer: 1,
    explanation:
      'Modular arithmetic prevents integer overflow for large computations. Common in interviews: "return answer mod 10^9+7". Used in cryptography (RSA), hashing, and handling large factorials/combinations.',
  },
  {
    id: 'mc5',
    question: 'What math concepts appear most in interviews?',
    options: [
      'Calculus',
      'Primes, GCD/LCM, modular arithmetic, combinations, basic geometry (distance, area)',
      'Advanced algebra',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Common interview math: 1) Prime checking/factorization, 2) GCD/LCM, 3) Modular arithmetic, 4) Combinations/permutations, 5) Basic geometry (distance formula, triangle area). Not calculus or advanced math.',
  },
];
