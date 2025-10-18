/**
 * Quiz questions for Common Algorithm Patterns section
 */

export const commonpatternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain common math patterns: digit manipulation, sum of multiples, power of numbers.',
    sampleAnswer:
      'Digit manipulation: extract digits with n % 10 (last digit) and n // 10 (remove last digit). Reverse number: result = result×10 + digit. Count digits: log10(n) + 1. Sum of digits: extract and sum. Sum of multiples: sum of multiples of k up to n is k×(1+2+...+m) where m = n//k. Use formula m×(m+1)/2. Power checks: power of 2 is (n & (n-1)) == 0, power of k needs log_k(n) to be integer. For example, reverse 123: take 3, result=3; take 2, result=32; take 1, result=321. Sum multiples of 3 up to 10: 3+6+9 = 3×(1+2+3) = 3×6 = 18. These patterns appear in many problems.',
    keyPoints: [
      'Digits: n%10 for last, n//10 to remove',
      'Reverse: result = result×10 + digit',
      'Sum multiples: k×m×(m+1)/2 where m=n//k',
      'Power of 2: n & (n-1) == 0',
      'Common in: number manipulation problems',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the sqrt(x) problem. How do you implement integer square root efficiently?',
    sampleAnswer:
      'Integer square root: find largest integer k where k² ≤ x. Approach 1: linear search O(sqrt(x)) - too slow. Approach 2: binary search O(log x) - efficient. Search range [0, x], check if mid² ≤ x. For example, sqrt(8): try mid=4, 16>8, search [0,3]; try mid=1, 1≤8, search [2,3]; try mid=2, 4≤8, search [3,3]; try mid=3, 9>8, answer is 2. Careful: mid² can overflow, use mid ≤ x/mid instead. Approach 3: Newton method (x_new = (x + n/x)/2), converges fast O(log log x). For coding interviews, binary search is standard. Key: handle overflow, correct boundaries. This is classic binary search application on answer space.',
    keyPoints: [
      'Find largest k where k² ≤ x',
      'Binary search on [0, x], O(log x)',
      'Check mid² ≤ x, avoid overflow',
      'Alternative: Newton method O(log log x)',
      'Classic binary search on answer',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through modular arithmetic. Why do we need it and how to apply it correctly?',
    sampleAnswer:
      'Modular arithmetic: operations under modulo m. Why needed? Prevent overflow in large number calculations, problem requirements (answer modulo 10^9+7). Properties: (a+b)%m = ((a%m)+(b%m))%m, (a-b)%m = ((a%m)-(b%m)+m)%m (add m to handle negative), (a×b)%m = ((a%m)×(b%m))%m. For division: use modular inverse. For example, compute n! % m: instead of n! then modulo (overflow), do: result = 1; for i in 1 to n: result = (result × i) % m. This keeps numbers bounded. Subtraction example: (5-8)%3 = (5%3-8%3+3)%3 = (2-2+3)%3 = 3%3 = 0 (correct), but (5-8)%3 = -3%3 could be -0 or 0 depending on language. Key: apply modulo at each step, handle negatives carefully.',
    keyPoints: [
      'Operations under modulo m',
      'Why: prevent overflow, problem requirement',
      'Properties: (a op b) % m = ((a%m) op (b%m)) % m',
      'Subtraction: add m to handle negatives',
      'Apply modulo at each step, keep bounded',
    ],
  },
];
