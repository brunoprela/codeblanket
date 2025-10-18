/**
 * Multiple choice questions for Number Theory Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const numbertheoryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How do you check if a number is prime?',
    options: [
      'Try all numbers',
      'Trial division up to √n - check divisibility by 2 and odd numbers',
      'Random',
      'Cannot check',
    ],
    correctAnswer: 1,
    explanation:
      'Prime check: if divisible by any number from 2 to √n, not prime. Only need √n because factors come in pairs. Optimization: check 2, then odd numbers 3,5,7... O(√n) time.',
  },
  {
    id: 'mc2',
    question: 'What is the Sieve of Eratosthenes?',
    options: [
      'Sorting algorithm',
      'Find all primes up to n by iteratively marking multiples of each prime as composite',
      'Random',
      'Search algorithm',
    ],
    correctAnswer: 1,
    explanation:
      'Sieve: create boolean array, mark composites. For each prime p, mark p², p²+p, p²+2p... as composite. O(n log log n) time. Efficient for finding many primes.',
  },
  {
    id: 'mc3',
    question: 'What is GCD and how do you compute it efficiently?',
    options: [
      'Greatest common divisor, use loops',
      'GCD(a,b) = largest number dividing both. Use Euclidean algorithm: GCD(a,b) = GCD(b, a mod b)',
      'Random',
      'Trial division',
    ],
    correctAnswer: 1,
    explanation:
      'GCD: largest number dividing both a and b. Euclidean algorithm: GCD(a,b) = GCD(b, a%b), base case: GCD(a,0)=a. O(log min(a,b)) time. LCM(a,b) = a*b/GCD(a,b).',
  },
  {
    id: 'mc4',
    question: 'What is modular arithmetic and why use it?',
    options: [
      'Random math',
      'Arithmetic with remainders - (a+b)%m = ((a%m) + (b%m))%m - prevents overflow',
      'Division',
      'Not useful',
    ],
    correctAnswer: 1,
    explanation:
      'Modular arithmetic: operations with mod m. Properties: (a+b)%m = ((a%m)+(b%m))%m, (a*b)%m = ((a%m)*(b%m))%m. Prevents overflow in large computations. Common: "return answer mod 10^9+7".',
  },
  {
    id: 'mc5',
    question: 'How do you compute power efficiently with modulo?',
    options: [
      'Loop multiplication',
      'Modular exponentiation: a^n mod m using binary exponentiation in O(log n)',
      'Random',
      'Cannot do',
    ],
    correctAnswer: 1,
    explanation:
      'Modular exponentiation: compute a^n mod m in O(log n) using binary representation. If n even: (a^(n/2))² mod m. If odd: a * (a^(n-1)) mod m. Apply mod at each step to prevent overflow.',
  },
];
