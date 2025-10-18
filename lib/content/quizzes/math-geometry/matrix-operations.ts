/**
 * Quiz questions for Matrix Manipulation section
 */

export const matrixoperationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain matrix rotation (90 degrees clockwise). What is the in-place approach?',
    sampleAnswer:
      'Rotate 90° clockwise: (i,j) → (j, n-1-i). Approach 1: create new matrix, copy with transformation. O(n²) time and space. Approach 2 (in-place): transpose (swap (i,j) with (j,i)), then reverse each row. O(n²) time, O(1) space. For example, [[1,2,3],[4,5,6],[7,8,9]]: transpose → [[1,4,7],[2,5,8],[3,6,9]], reverse rows → [[7,4,1],[8,5,2],[9,6,3]]. Why it works: transpose swaps across diagonal, reverse horizontally completes clockwise rotation. For 90° counter-clockwise: transpose then reverse each column (or reverse rows then transpose). For 180°: reverse rows and reverse each row (or call 90° twice). In-place is space-efficient but modifies original. Key: understand transformation formula.',
    keyPoints: [
      'Clockwise 90°: (i,j) → (j, n-1-i)',
      'In-place: transpose, then reverse rows',
      'O(n²) time, O(1) space',
      'Counter-clockwise: transpose, reverse columns',
      'Understand: transformation formula',
    ],
  },
  {
    id: 'q2',
    question: 'Describe spiral matrix traversal. How do you track boundaries?',
    sampleAnswer:
      'Spiral: traverse outer ring, then inner rings recursively. Track four boundaries: top, bottom, left, right. Algorithm: 1) Traverse top row left→right (left to right), increment top. 2) Traverse right column top→bottom (top+1 to bottom), decrement right. 3) Traverse bottom row right→left (right to left, if top <= bottom), decrement bottom. 4) Traverse left column bottom→top (bottom-1 to top, if left <= right), increment left. 5) Repeat until boundaries cross. For [[1,2,3],[4,5,6],[7,8,9]]: top row 1,2,3; right col 6,9; bottom row 8,7; left col 4; center 5. Result: [1,2,3,6,9,8,7,4,5]. Tricky parts: checking boundaries before each traverse, handling single row/column. O(m×n) time, O(1) space (excluding result).',
    keyPoints: [
      'Traverse: outer rings to inner',
      'Track: top, bottom, left, right boundaries',
      'Order: right, down, left, up',
      'Check boundaries before each direction',
      'O(m×n) time, handle single row/column',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through matrix multiplication. What is the complexity and why?',
    sampleAnswer:
      'Matrix multiplication: A (m×n) × B (n×p) = C (m×p). Each element C[i][j] = sum of A[i][k] × B[k][j] for k=0 to n-1. Three nested loops: for i (m), for j (p), for k (n). Complexity: O(m×n×p). For square matrices (n×n): O(n³). For example, [[1,2],[3,4]] × [[5,6],[7,8]]: C[0][0] = 1×5 + 2×7 = 19, C[0][1] = 1×6 + 2×8 = 22, etc. Result: [[19,22],[43,50]]. Note: matrix multiplication is not commutative (A×B ≠ B×A). Advanced algorithms: Strassen O(n^2.807), Coppersmith-Winograd O(n^2.376) but impractical. In practice, use standard O(n³) or libraries with hardware optimization. Used for: transformations, graph algorithms (adjacency matrix powers).',
    keyPoints: [
      'A(m×n) × B(n×p) = C(m×p)',
      'C[i][j] = sum A[i][k] × B[k][j]',
      'Three nested loops: O(m×n×p)',
      'Square: O(n³), not commutative',
      'Uses: transformations, graph powers',
    ],
  },
];
