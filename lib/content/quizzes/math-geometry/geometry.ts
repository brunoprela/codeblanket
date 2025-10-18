/**
 * Quiz questions for Coordinate Geometry section
 */

export const geometryQuiz = [
  {
    id: 'q1',
    question:
      'Compare Euclidean distance vs Manhattan distance. When do you use each?',
    sampleAnswer:
      'Euclidean distance: straight-line distance, sqrt((x2-x1)² + (y2-y1)²). Represents physical distance "as crow flies". Manhattan distance: sum of absolute differences, |x2-x1| + |y2-y1|. Represents grid distance with only horizontal/vertical moves. For (0,0) to (3,4): Euclidean = sqrt(9+16) = 5, Manhattan = 3+4 = 7. Use Euclidean when: actual physical distance matters, diagonals allowed, geometry problems. Use Manhattan when: grid-based movement (taxicab), no diagonal moves, easier to compute (no sqrt). For example, distance between cities (Euclidean), robot on grid (Manhattan), chess king moves (Chebyshev = max(|dx|, |dy|)). Manhattan is faster (no sqrt) and matches grid movement constraints.',
    keyPoints: [
      'Euclidean: sqrt((dx)² + (dy)²) - straight line',
      'Manhattan: |dx| + |dy| - grid with h/v only',
      'Euclidean: physical distance, diagonals',
      'Manhattan: grid movement, no diagonals',
      'Manhattan faster: no sqrt computation',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how to find if a point is inside a triangle. What approaches exist?',
    sampleAnswer:
      'Three approaches: 1) Area method: compute area of triangle ABC and areas of PAB, PBC, PCA. If sum of sub-areas equals total, P is inside. 2) Barycentric coordinates: express P as weighted sum of A,B,C. If all weights in [0,1], P is inside. 3) Cross product: check if P is on same side of all three edges. For area method: area using cross product |AB × AC|/2. For example, triangle (0,0), (4,0), (0,3), point (1,1): area(ABC)=6, area(PAB)+area(PBC)+area(PCA)=6 → inside. If sum>6, outside. Time O(1) for all methods. Area method is simplest. Edge case: point on edge (use <= for inclusive). Used for: collision detection, mesh rendering, computational geometry.',
    keyPoints: [
      'Area method: sum sub-areas = total?',
      'Barycentric: weights all in [0,1]?',
      'Cross product: same side of all edges?',
      'All O(1), area method simplest',
      'Uses: collision, rendering, geometry',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe closest pair of points problem. What is the efficient algorithm?',
    sampleAnswer:
      'Problem: given n points, find pair with minimum distance. Brute force: check all pairs O(n²). Efficient: divide-and-conquer O(n log n). Algorithm: 1) Sort by x-coordinate. 2) Divide points into left and right halves. 3) Recursively find closest pairs in each half: dL and dR. 4) Let d = min(dL, dR). 5) Check strip of width 2d around dividing line for closer pairs (points within d of line). 6) Sort strip by y, check next 7 points for each (geometric proof: at most 7 candidates). For example, 8 points: divide into 4+4, solve recursively, merge with strip check. Why O(n log n)? Divide: O(n log n) sort, conquer: T(n) = 2T(n/2) + O(n), gives O(n log n). Key insight: only need to check 7 neighbors in strip.',
    keyPoints: [
      'Brute force: O(n²) check all pairs',
      'Efficient: O(n log n) divide-and-conquer',
      'Divide by x, solve halves, check strip',
      'Strip: only 7 neighbors per point needed',
      'Master theorem: T(n) = 2T(n/2) + O(n) = O(n log n)',
    ],
  },
];
