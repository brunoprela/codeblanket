/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize Math & Geometry problems in interviews? What signals these?',
    sampleAnswer:
      'Keywords: "prime", "GCD", "factorial", "modulo", "rotate", "matrix", "distance", "point", "angle", "area", "digit", "power", "mathematical". Patterns: 1) Number properties (even/odd, prime, divisibility). 2) Coordinate problems (points, distances, shapes). 3) Matrix operations (rotation, traversal, multiplication). 4) Sequences (Fibonacci, factorial, Catalan). 5) Combinatorics (permutations, combinations). 6) Modular arithmetic. For example, "count primes up to n" → number theory. "Rotate matrix 90 degrees" → matrix manipulation. "Find closest pair of points" → computational geometry. "Generate valid parentheses" → Catalan. Signals: math terminology, coordinates, geometric shapes, modulo requirements. Often simpler with right formula than complex algorithm.',
    keyPoints: [
      'Keywords: prime, GCD, rotate, matrix, distance, modulo',
      'Patterns: number properties, coordinates, sequences',
      'Examples: primes, matrix rotate, closest points',
      'Signals: math terms, coordinates, shapes',
      'Often: formula simpler than algorithm',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your interview approach for Math & Geometry problems from recognition to solution.',
    sampleAnswer:
      'First, recognize math/geometry from keywords (prime, matrix, distance, modulo). Second, recall relevant formulas or properties (GCD algorithm, rotation formula, distance metrics). Third, consider edge cases (0, negative, overflow, precision). Fourth, choose efficient algorithm (Sieve vs trial, iteration vs recursion). Fifth, implement carefully with bounds checking. Sixth, test with examples including edges. Finally, analyze complexity and discuss alternatives. For example, "rotate matrix": recognize as matrix manipulation, recall transpose+reverse formula, implement in-place O(n²) O(1), test with 3×3 and edge case 1×1, discuss transpose-then-reverse vs four-way swap. Show: pattern recognition, formula knowledge, implementation care, edge case awareness.',
    keyPoints: [
      'Recognize: keywords, math patterns',
      'Recall: formulas, properties, algorithms',
      'Consider: edge cases, overflow, precision',
      'Choose: efficient approach (Sieve, iteration)',
      'Test: examples, edges, analyze complexity',
      'Discuss alternatives',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in Math & Geometry problems? How do you avoid them?',
    sampleAnswer:
      'First: integer overflow (factorial, power grows fast). Second: floating-point precision (compare with epsilon, not ==). Third: modulo errors (forgetting at each step, negative modulo). Fourth: off-by-one in coordinates or loops. Fifth: wrong formula (transpose vs rotation). Sixth: edge cases (0, 1, negative, MAX_INT). Seventh: inefficient algorithm (trial division when Sieve better). My strategy: 1) Use long/BigInteger for large numbers. 2) Compare floats with abs (a-b) < epsilon. 3) Apply modulo at each operation. 4) Draw diagrams for geometry. 5) Double-check formulas. 6) Test: 0, 1, negative, large values. 7) Know complexity of algorithms. For example, factorial: never compute n! then modulo, do modulo at each multiply.',
    keyPoints: [
      'Mistakes: overflow, precision, modulo, off-by-one',
      'Formula errors, edge cases, slow algorithms',
      'Avoid: long types, epsilon, modulo early',
      'Test thoroughly: 0, 1, negative, large',
      'Know: formulas, complexities, alternatives',
      'Think mathematically before coding',
    ],
  },
];
