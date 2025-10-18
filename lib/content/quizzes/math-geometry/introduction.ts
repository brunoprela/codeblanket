/**
 * Quiz questions for Introduction to Math & Geometry section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what Math & Geometry problems are in algorithms. How do they differ from typical data structure problems?',
    sampleAnswer:
      'Math & Geometry problems focus on mathematical properties, formulas, and spatial relationships rather than data structure manipulation. Categories: 1) Number theory (primes, GCD, modular arithmetic), 2) Matrix operations (rotation, traversal), 3) Geometry (points, lines, areas), 4) Combinatorics (permutations, combinations). Different because: require mathematical insight not algorithmic patterns, often closed-form solutions exist, heavy use of formulas and properties. For example, "rotate matrix 90 degrees" is manipulation formula, not search/sort. "Check if prime" uses math properties (trial division, Sieve). "Count ways to arrange" is combinatorics. "Find closest points" is computational geometry. These test mathematical thinking alongside coding. Success requires: knowing formulas, recognizing patterns, avoiding overflow, precision issues.',
    keyPoints: [
      'Focus on: math properties, formulas, spatial relationships',
      'Categories: number theory, matrix, geometry, combinatorics',
      'Different: mathematical insight vs data structures',
      'Examples: primes, matrix rotation, point distances',
      'Requires: formulas, patterns, precision handling',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Math & Geometry problems to other algorithm categories. When do you recognize them?',
    sampleAnswer:
      'Recognition signals: problems mention numbers, coordinates, shapes, angles, mathematical operations. Keywords: "prime", "GCD", "factorial", "modulo", "matrix", "rotate", "point", "distance", "area", "angle". Unlike: graphs (edges/vertices), trees (hierarchical), arrays (sequential). Math problems are: self-contained calculations, formula-based, require mathematical background. For example, "shortest path" is graph. "Closest pair of points" is geometry. "Rotate array" is array manipulation. "Rotate matrix" is math/geometry. "Count primes up to n" is number theory. "Fibonacci" is sequences. When I see: coordinates (x,y), mathematical terms, geometric shapes, modulo operations, factorials, I think Math & Geometry. Often easier to solve with right formula.',
    keyPoints: [
      'Keywords: prime, GCD, modulo, matrix, point, angle',
      'Self-contained calculations, formula-based',
      'vs Graphs/Trees: no edges/nodes',
      'Examples: primes, rotation, distances, sequences',
      'Know formulas → easy solve',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common pitfalls in Math & Geometry problems? How do you avoid them?',
    sampleAnswer:
      'First: integer overflow (factorial, power grows fast). Second: floating-point precision (0.1 + 0.2 != 0.3). Third: modulo arithmetic errors (negative modulo, order of operations). Fourth: off-by-one in coordinates. Fifth: edge cases (0, negative, MAX_INT). Sixth: inefficient algorithms (trial division vs Sieve for primes). Seventh: missing mathematical insights (brute force when formula exists). Avoidance: 1) Use long/BigInteger for large numbers. 2) Compare floats with epsilon tolerance. 3) Keep numbers bounded with modulo early. 4) Draw diagrams for geometry. 5) Test: 0, 1, negative, large. 6) Study common formulas and optimizations. 7) Think mathematically before coding. For example, computing n! naively overflows; use modulo at each step.',
    keyPoints: [
      'Pitfalls: overflow, precision, modulo errors',
      'Edge cases: 0, negative, max values',
      'Inefficiency: brute force vs formulas',
      'Solutions: long types, epsilon, modulo early',
      'Test thoroughly, think math first',
    ],
  },
];
