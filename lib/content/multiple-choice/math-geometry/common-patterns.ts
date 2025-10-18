/**
 * Multiple choice questions for Common Algorithm Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const commonpatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the mathematical simplification pattern?',
    options: [
      'Random',
      'Derive closed-form formula from recurrence - O(1) instead of O(n)',
      'Loop optimization',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Mathematical simplification: recognize pattern and derive direct formula. Example: sum 1..n = n*(n+1)/2 instead of loop. Arithmetic/geometric series have closed forms. O(1) vs O(n).',
  },
  {
    id: 'mc2',
    question: 'What is the GCD pattern?',
    options: [
      'Random',
      'Use GCD for divisibility, simplification, LCM. Euclidean algorithm O(log min(a,b))',
      'Sorting',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'GCD pattern: problems involving divisibility, reducing fractions, finding LCM. LCM(a,b) = a*b/GCD(a,b). Euclidean algorithm: GCD(a,b) = GCD(b, a%b). O(log n) time.',
  },
  {
    id: 'mc3',
    question: 'What is the digit manipulation pattern?',
    options: [
      'String conversion',
      'Extract digits: n%10 for last, n//10 to remove. Build number: result*10+digit',
      'Random',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Digit manipulation: extract last digit with n%10, remove with n//10. Build number: result = result*10 + digit. Check palindrome, reverse number, sum digits without string conversion.',
  },
  {
    id: 'mc4',
    question: 'What is the sieve pattern?',
    options: [
      'Random',
      'Precompute properties for range [1..n] - primes, divisors. Trade space for time',
      'Sorting',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Sieve pattern: precompute for all numbers up to n. Sieve of Eratosthenes for primes O(n log log n). Similar for smallest prime factor, divisor count. O(n) space, efficient for multiple queries.',
  },
  {
    id: 'mc5',
    question: 'What is the modular arithmetic pattern?',
    options: [
      'Random',
      'Apply mod after each operation to prevent overflow: (a op b) % m',
      'Final mod only',
      'No pattern',
    ],
    correctAnswer: 1,
    explanation:
      'Modular arithmetic: apply mod after each operation. (a+b)%m = ((a%m)+(b%m))%m. (a*b)%m = ((a%m)*(b%m))%m. Prevents overflow. Common: "return answer mod 10^9+7".',
  },
];
