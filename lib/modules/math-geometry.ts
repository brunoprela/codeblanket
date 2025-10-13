import { Module } from '@/lib/types';

export const mathGeometryModule: Module = {
  id: 'math-geometry',
  title: 'Math & Geometry',
  description:
    'Master mathematical algorithms and geometric problem-solving techniques.',
  icon: '📐',
  timeComplexity: 'Varies by algorithm',
  spaceComplexity: 'Usually O(1) to O(n)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Math & Geometry',
      content: `**Math and Geometry** problems test your ability to recognize patterns, apply mathematical principles, and think creatively about spatial relationships.

**Why Learn Math & Geometry?**
- **Pattern recognition**: Identify mathematical relationships
- **Optimization**: Find efficient solutions using math properties
- **Real-world applications**: Used in graphics, games, simulations
- **Interview staples**: Common in coding interviews
- **Foundation**: Builds problem-solving intuition

**Common Problem Types:**
- Number theory (primes, divisors, GCD/LCM)
- Matrix manipulation (rotation, spiral, search)
- Coordinate geometry (distances, areas)
- Combinatorics (permutations, combinations)
- Mathematical sequences (Fibonacci, factorials)
- Modular arithmetic
- Angle and shape calculations`,
      quiz: [
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
      ],
    },
    {
      id: 'number-theory',
      title: 'Number Theory Fundamentals',
      content: `**Essential Number Theory Concepts:**

**1. Prime Numbers**
- Only divisible by 1 and itself
- Sieve of Eratosthenes for finding all primes up to n

**2. GCD (Greatest Common Divisor)**
- Euclidean Algorithm: O(log(min(a,b)))
\`\`\`python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
\`\`\`

**3. LCM (Least Common Multiple)**
- Formula: LCM(a,b) = (a × b) / GCD(a,b)

**4. Factorization**
- Break number into prime factors
- Check divisors up to sqrt(n)

**5. Power and Exponentiation**
- Fast exponentiation: O(log n)
- Use for large powers efficiently

**6. Modular Arithmetic**
- (a + b) % m = ((a % m) + (b % m)) % m
- (a × b) % m = ((a % m) × (b % m)) % m
- Important for handling large numbers`,
      codeExample: `# Number theory essentials

def gcd(a: int, b: int) -> int:
    """Greatest Common Divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least Common Multiple."""
    return (a * b) // gcd(a, b)


def is_prime(n: int) -> bool:
    """Check if n is prime. O(sqrt(n))"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def prime_factorization(n: int) -> dict:
    """Get prime factors with their counts."""
    factors = {}
    
    # Check for 2s
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    
    # Check odd numbers
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2
    
    if n > 2:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def fast_power(base: int, exp: int, mod: int = None) -> int:
    """Fast exponentiation. O(log exp)"""
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod if mod else result * base
        exp //= 2
        base = (base * base) % mod if mod else base * base
    
    return result


# Examples
print(f"GCD(48, 18) = {gcd(48, 18)}")  # 6
print(f"LCM(12, 15) = {lcm(12, 15)}")  # 60
print(f"Is 17 prime? {is_prime(17)}")  # True
print(f"Factors of 12: {prime_factorization(12)}")  # {2: 2, 3: 1}
print(f"2^10 mod 1000 = {fast_power(2, 10, 1000)}")  # 24`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain prime number checking. Compare trial division vs Sieve of Eratosthenes approaches.',
          sampleAnswer:
            'Prime: number divisible only by 1 and itself. Trial division: check if n divisible by any number from 2 to sqrt(n). O(sqrt(n)) per check. Sieve of Eratosthenes: find all primes up to n by marking multiples. O(n log log n) for all primes up to n. Use trial division when: checking single number, n is small. Use Sieve when: need many primes, precompute up to limit. For example, check if 17 prime: test 2,3,4 (stop at sqrt(17)≈4) - none divide, so prime. For primes up to 100: Sieve marks multiples of 2 (4,6,8,...), then 3 (6,9,12,...), etc. Remaining unmarked are primes. Trial division: O(sqrt(n)) per check. Sieve: O(n log log n) total for all primes ≤ n. For multiple queries, Sieve precomputation wins.',
          keyPoints: [
            'Prime: divisible by 1 and self only',
            'Trial division: O(sqrt(n)) per check',
            'Sieve: O(n log log n) for all primes ≤ n',
            'Trial when: single check, small n',
            'Sieve when: multiple queries, precompute',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe GCD (Greatest Common Divisor) and Euclidean algorithm. Why is it efficient?',
          sampleAnswer:
            'GCD: largest number dividing both a and b. Euclidean algorithm: gcd(a, b) = gcd(b, a % b), base case gcd(a, 0) = a. Works because: any divisor of a and b also divides (a - kb) for any k, including a % b. Repeat until remainder 0. For example, gcd(48, 18): 48 % 18 = 12, gcd(18, 12); 18 % 12 = 6, gcd(12, 6); 12 % 6 = 0, gcd(6, 0) = 6. Time complexity: O(log min(a,b)) - each step reduces numbers by at least half (Fibonacci numbers are worst case). Space: O(1) iterative or O(log n) recursive. Why efficient? Much faster than factorization. Used for: simplifying fractions, LCM (lcm = a*b/gcd), modular inverse. This is one of oldest algorithms (300 BC).',
          keyPoints: [
            'GCD: largest divisor of both numbers',
            'Euclidean: gcd(a, b) = gcd(b, a % b)',
            'O(log min(a,b)) - very fast',
            'Each step reduces by at least half',
            'Uses: fractions, LCM, modular inverse',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain fast power (exponentiation by squaring). Why is it O(log n) instead of O(n)?',
          sampleAnswer:
            'Fast power computes a^n efficiently by squaring. Key insight: a^n = (a^(n/2))^2 if n even, a^n = a × (a^((n-1)/2))^2 if n odd. Instead of n multiplications (a×a×...×a), do log n by repeatedly squaring. For example, 2^10: 2^10 = (2^5)^2, 2^5 = 2×(2^2)^2, 2^2 = (2^1)^2, 2^1 = 2. Total 4 multiplications vs 10. With modulo (for large results): compute (a^n) % m by taking modulo at each step to keep numbers bounded. Algorithm: start with result=1, while n>0: if n odd, result *= a; a *= a; n /= 2. Why O(log n)? Each iteration halves n. Used for: large exponents, modular exponentiation (RSA cryptography), matrix exponentiation (Fibonacci).',
          keyPoints: [
            'Compute a^n by repeated squaring',
            'O(log n) vs O(n) naive multiplication',
            'Even: (a^(n/2))^2, Odd: a × (a^((n-1)/2))^2',
            'With modulo: keep numbers bounded',
            'Uses: large exponents, crypto, Fibonacci',
          ],
        },
      ],
    },
    {
      id: 'matrix-operations',
      title: 'Matrix Manipulation',
      content: `**Common Matrix Operations:**

**1. Matrix Traversal Patterns:**
- Row by row: Standard nested loops
- Column by column: Reverse loop order
- Diagonal: i+j or i-j patterns
- Spiral: Layer by layer from outside in
- Snake: Alternate row directions

**2. Matrix Rotation:**
- **90° Clockwise**: Transpose + Reverse each row
- **90° Counter-clockwise**: Reverse each row + Transpose
- **180°**: Reverse rows + Reverse columns

**3. Matrix Transpose:**
- Swap matrix[i][j] with matrix[j][i]
- Only need to process upper triangle

**4. Matrix Spiral:**
- Process layer by layer
- Track top, bottom, left, right boundaries
- Move in order: right → down → left → up

**5. In-Place Operations:**
- Many matrix problems require O(1) space
- Use clever swapping and boundary tracking`,
      codeExample: `# Matrix operation examples

def rotate_90_clockwise(matrix):
    """Rotate matrix 90° clockwise in-place."""
    n = len(matrix)
    
    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()


def spiral_order(matrix):
    """Return elements in spiral order."""
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Move right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Move down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Move left (if still rows)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Move up (if still columns)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result


def diagonal_traverse(matrix):
    """Traverse matrix diagonally."""
    if not matrix:
        return []
    
    m, n = len(matrix), len(matrix[0])
    result = []
    
    # Process all diagonals
    for d in range(m + n - 1):
        intermediate = []
        
        # Find start position
        row = 0 if d < n else d - n + 1
        col = d if d < n else n - 1
        
        # Collect diagonal
        while row < m and col >= 0:
            intermediate.append(matrix[row][col])
            row += 1
            col -= 1
        
        # Reverse alternate diagonals
        if d % 2 == 0:
            result.extend(intermediate[::-1])
        else:
            result.extend(intermediate)
    
    return result


# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Spiral: {spiral_order(matrix)}")
# Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]`,
      quiz: [
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
          question:
            'Describe spiral matrix traversal. How do you track boundaries?',
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
      ],
    },
    {
      id: 'geometry',
      title: 'Coordinate Geometry',
      content: `**Essential Geometric Concepts:**

**1. Distance Formula:**
- Euclidean distance: sqrt((x2-x1)² + (y2-y1)²)
- Manhattan distance: |x2-x1| + |y2-y1|

**2. Area Calculations:**
- **Triangle**: 0.5 × base × height
- **Rectangle**: width × height
- **Circle**: π × r²
- **Polygon** (Shoelace formula):
\`\`\`python
def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2
\`\`\`

**3. Angle Calculations:**
- Dot product: a·b = |a||b|cos(θ)
- Cross product: Determines orientation
- Slope: (y2-y1)/(x2-x1)

**4. Point-Line Relationships:**
- Point on line segment
- Shortest distance from point to line
- Line intersection

**5. Convex Hull:**
- Find smallest convex polygon containing all points
- Graham scan: O(n log n)
- Jarvis march: O(nh) where h = hull size`,
      codeExample: `import math

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def manhattan_distance(p1, p2):
    """Calculate Manhattan distance."""
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])


def triangle_area(p1, p2, p3):
    """Calculate triangle area using cross product."""
    return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
               (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2


def is_point_in_triangle(pt, p1, p2, p3):
    """Check if point is inside triangle using areas."""
    area_total = triangle_area(p1, p2, p3)
    area1 = triangle_area(pt, p2, p3)
    area2 = triangle_area(p1, pt, p3)
    area3 = triangle_area(p1, p2, pt)
    
    return abs(area_total - (area1 + area2 + area3)) < 1e-10


def point_to_line_distance(point, line_start, line_end):
    """Shortest distance from point to line segment."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle zero-length line
    if dx == 0 and dy == 0:
        return euclidean_distance(point, line_start)
    
    # Parameter t for closest point on line
    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)))
    
    # Closest point on line
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return euclidean_distance(point, (closest_x, closest_y))


def rotate_point(point, angle, center=(0, 0)):
    """Rotate point around center by angle (in radians)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Translate to origin
    x = point[0] - center[0]
    y = point[1] - center[1]
    
    # Rotate
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    
    # Translate back
    return (new_x + center[0], new_y + center[1])


# Examples
p1 = (0, 0)
p2 = (3, 4)
print(f"Euclidean distance: {euclidean_distance(p1, p2)}")  # 5.0
print(f"Manhattan distance: {manhattan_distance(p1, p2)}")  # 7`,
      quiz: [
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
      ],
    },
    {
      id: 'combinatorics',
      title: 'Combinatorics and Sequences',
      content: `**Common Combinatorial Concepts:**

**1. Permutations:**
- Order matters
- n! = n × (n-1) × ... × 1
- nPr = n!/(n-r)! for r items from n

**2. Combinations:**
- Order doesn't matter
- nCr = n!/(r!(n-r)!)
- Pascal's triangle: C(n,r) = C(n-1,r-1) + C(n-1,r)

**3. Fibonacci Sequence:**
- F(n) = F(n-1) + F(n-2)
- F(0) = 0, F(1) = 1
- Many DP problems use Fibonacci pattern

**4. Factorials:**
- n! grows very fast
- Often need modular arithmetic
- Trailing zeros in n! = count of 5s in factors

**5. Catalan Numbers:**
- C(n) = (2n)!/(n!(n+1)!)
- Count valid parentheses, BSTs, etc.
- C(n) = C(0)C(n-1) + C(1)C(n-2) + ... + C(n-1)C(0)`,
      codeExample: `# Combinatorics implementations

def factorial(n: int) -> int:
    """Calculate n! iteratively."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def permutations(n: int, r: int) -> int:
    """Calculate nPr = n!/(n-r)!"""
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result


def combinations(n: int, r: int) -> int:
    """Calculate nCr = n!/(r!(n-r)!)"""
    if r > n - r:
        r = n - r
    
    result = 1
    for i in range(r):
        result *= (n - i)
        result //= (i + 1)
    
    return result


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number. O(n) time, O(1) space."""
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


def fibonacci_matrix(n: int) -> int:
    """Calculate Fibonacci using matrix exponentiation. O(log n)"""
    def matrix_mult(a, b):
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]
        ]
    
    def matrix_power(mat, n):
        if n == 1:
            return mat
    if n % 2 == 0:
            half = matrix_power(mat, n // 2)
            return matrix_mult(half, half)
        return matrix_mult(mat, matrix_power(mat, n - 1))
    
    if n <= 1:
        return n
    
    base = [[1, 1], [1, 0]]
    result = matrix_power(base, n)
    return result[0][1]


def catalan(n: int) -> int:
    """Calculate nth Catalan number."""
    if n <= 1:
        return 1
    
    # Using DP
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    
    return dp[n]


# Examples
print(f"5! = {factorial(5)}")  # 120
print(f"5P3 = {permutations(5, 3)}")  # 60
print(f"5C3 = {combinations(5, 3)}")  # 10
print(f"Fib(10) = {fibonacci(10)}")  # 55
print(f"Catalan(4) = {catalan(4)}")  # 14`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain permutations vs combinations. How do you compute each?',
          sampleAnswer:
            'Permutations: order matters. nPr = n!/(n-r)! ways to arrange r items from n. Combinations: order does not matter. nCr = n!/(r!(n-r)!) ways to choose r from n. For example, 3 items {A,B,C}, choose 2: permutations are AB, BA, AC, CA, BC, CB (6 = 3P2). Combinations are AB, AC, BC (3 = 3C2). Compute: use formula with factorials, but watch overflow. Better: nCr = nC(n-r) (symmetry), compute iteratively to avoid large factorials. Pascal triangle: nCr = (n-1)C(r-1) + (n-1)Cr. For example, 5C3 = 5!/(3!2!) = 120/(6×2) = 10. Applications: counting problems, probability, choosing teams, generating combinations.',
          keyPoints: [
            'Permutations: order matters, nPr = n!/(n-r)!',
            'Combinations: order irrelevant, nCr = n!/(r!(n-r)!)',
            'Compute iteratively to avoid overflow',
            'Pascal triangle: nCr = (n-1)C(r-1) + (n-1)Cr',
            'Uses: counting, probability, team selection',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe Fibonacci sequence. What are different ways to compute it?',
          sampleAnswer:
            'Fibonacci: F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1. Sequence: 0,1,1,2,3,5,8,13,... Approach 1: recursive F(n) = F(n-1) + F(n-2), O(2^n) time (exponential, slow). Approach 2: memoization (top-down DP), O(n) time and space. Approach 3: iteration (bottom-up DP), O(n) time, O(1) space. Approach 4: matrix exponentiation, O(log n) time. Approach 5: closed-form (Binet formula), O(1) but precision issues. For example, F(10): recursive does 177 calls, iteration does 10 steps. Matrix method: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n. Use fast power for O(log n). Best: iteration for moderate n, matrix for very large n.',
          keyPoints: [
            'F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1',
            'Recursive: O(2^n) - slow',
            'Iteration: O(n) time, O(1) space - good',
            'Matrix exponentiation: O(log n) - best for large n',
            'Binet formula: O(1) but precision issues',
          ],
        },
        {
          id: 'q3',
          question: 'Explain Catalan numbers. What problems do they solve?',
          sampleAnswer:
            'Catalan numbers: C(n) = (2n)! / ((n+1)!n!) = C(n-1) × 2(2n-1)/(n+1). Sequence: 1,1,2,5,14,42,... Count structures with recursive nesting. Problems: 1) Number of valid parenthesis sequences (n pairs). 2) Number of BSTs with n nodes. 3) Number of ways to triangulate polygon with n+2 sides. 4) Number of paths in n×n grid (not crossing diagonal). 5) Number of binary trees with n nodes. For example, C(3) = 5: valid parentheses are ((())), (()()), (())(), ()(()), ()()(). BSTs with 3 nodes: 5 different structures. Recurrence: C(n) = sum C(i)×C(n-1-i) for i=0 to n-1. Compute iteratively with formula. Catalan appears in many combinatorial problems with nested or recursive structure.',
          keyPoints: [
            'C(n) = (2n)! / ((n+1)!n!), recursive structures',
            'Sequence: 1,1,2,5,14,42,...',
            'Problems: valid parentheses, BSTs, triangulations',
            'Recurrence: C(n) = sum C(i)×C(n-1-i)',
            'Appears in: nested, recursive, binary structures',
          ],
        },
      ],
    },
    {
      id: 'common-patterns',
      title: 'Common Algorithm Patterns',
      content: `**Pattern Recognition:**

**1. Happy Number / Cycle Detection:**
- Use Floyd's cycle detection (two pointers)
- Transform number repeatedly until cycle or target

**2. Power Function:**
- Fast exponentiation: O(log n)
- Handle negative exponents
- Consider overflow

**3. Digit Manipulation:**
- Extract digits: n % 10, n //= 10
- Reverse number: build from digits
- Sum of digit squares: for happy number

**4. Sieve Algorithms:**
- Sieve of Eratosthenes for primes
- Mark multiples as composite
- O(n log log n) time

**5. Mathematical Optimization:**
- Use math properties to reduce brute force
- Example: Check divisors only up to sqrt(n)
- Use formulas instead of loops when possible

**6. Modular Arithmetic:**
- Keep numbers bounded
- (a + b) % m = ((a % m) + (b % m)) % m
- Useful for large number problems`,
      quiz: [
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
      ],
    },
    {
      id: 'complexity',
      title: 'Time and Space Complexity',
      content: `**Common Complexities:**

**Number Theory:**
- GCD (Euclidean): O(log(min(a,b)))
- Prime check: O(sqrt(n))
- Sieve of Eratosthenes: O(n log log n)
- Factorization: O(sqrt(n))
- Fast exponentiation: O(log n)

**Matrix Operations:**
- Matrix traversal: O(m × n)
- Matrix transpose: O(m × n)
- Matrix rotation in-place: O(m × n), O(1) space
- Spiral traversal: O(m × n)

**Geometry:**
- Distance calculation: O(1)
- Point in polygon: O(n) for n vertices
- Convex hull (Graham scan): O(n log n)
- Line intersection: O(1)

**Combinatorics:**
- Factorial: O(n)
- Combinations (iterative): O(min(r, n-r))
- Fibonacci (iterative): O(n), O(1) space
- Fibonacci (matrix): O(log n)
- Catalan number: O(n²) with DP

**Space Optimization:**
- Most math problems can be solved with O(1) space
- Use iterative over recursive when possible
- Reuse variables instead of arrays`,
      quiz: [
        {
          id: 'q1',
          question:
            'Analyze complexity of math operations. Which are fast vs slow?',
          sampleAnswer:
            'Fast O(1): addition, subtraction, multiplication (constant-size integers), bitwise ops, simple comparisons. Moderate: division (slower than multiply), modulo (similar to division). Slow: square root (iterative methods), trigonometric functions, logarithms. For large numbers: fast power O(log n), factorial O(n) linear, GCD O(log min(a,b)), prime check O(sqrt(n)) trial division or O(1) with Sieve. Matrix operations: O(n³) for n×n. For example, adding two 32-bit ints is single CPU instruction O(1). Computing n! naively is n multiplications O(n). Sieve for primes up to n is O(n log log n). Key: understand what operations are expensive, optimize bottlenecks. In practice: addition fast, division moderate, sqrt slow.',
          keyPoints: [
            'Fast O(1): +, -, ×, bitwise',
            'Moderate: /, % (slower than multiply)',
            'Slow: sqrt, trig, log (iterative)',
            'Large numbers: factorial O(n), GCD O(log n)',
            'Matrix: O(n³) for n×n',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare different algorithms for same math problem (e.g., Fibonacci, factorial). When to use each?',
          sampleAnswer:
            'Fibonacci: recursive O(2^n) - never use, memoization O(n) - simple but space, iteration O(n) - standard, matrix O(log n) - large n. Factorial: loop O(n) - standard, cache with DP - reuse, modular at each step - prevent overflow. Prime check: trial division O(sqrt(n)) - single check, Sieve O(n log log n) - many queries. GCD: Euclidean O(log n) - always use, naive O(min(a,b)) - never. Choose based on: problem size, space constraints, query frequency. For example, Fibonacci up to 50: iteration works. For n=10^18: matrix exponentiation. For checking if 1000 numbers are prime: Sieve precomputation. For single GCD: Euclidean. Key: know complexity and constraints.',
          keyPoints: [
            'Fibonacci: iterate O(n) standard, matrix O(log n) large',
            'Factorial: loop O(n), modular at each step',
            'Primes: trial O(sqrt(n)) single, Sieve O(n log log n) many',
            'GCD: Euclidean O(log n) always',
            'Choose based on: size, space, frequency',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain space-time tradeoffs in math problems (e.g., Sieve vs trial division).',
          sampleAnswer:
            'Sieve of Eratosthenes: precompute all primes up to n, O(n log log n) time, O(n) space. Then answer "is p prime?" in O(1). Trial division: O(sqrt(n)) per query, O(1) space. Tradeoff: Sieve pays upfront cost and space for fast queries. Use Sieve when: many queries (amortized cost low), n is reasonable (< 10^7). Use trial when: few queries, n very large, memory limited. Similar: factorial precomputation O(n) space for O(1) query vs O(n) per computation. Memoization: O(n) space for O(1) lookup vs recomputing. The pattern: precompute (time+space) for fast queries, or compute on-demand (time only) for space efficiency. Choice depends on query frequency and memory constraints.',
          keyPoints: [
            'Sieve: O(n) space, O(1) query vs Trial: O(1) space, O(sqrt(n)) query',
            'Precompute (time+space) vs on-demand (time only)',
            'Sieve when: many queries, reasonable n',
            'Trial when: few queries, large n, memory limited',
            'Pattern: space for speed tradeoff',
          ],
        },
      ],
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognizing Math & Geometry Problems:**

**Keywords to watch for:**
- "Rotate" → Matrix rotation
- "Spiral" → Layer-by-layer traversal
- "Prime" → Number theory
- "Factorial", "Combinations" → Combinatorics
- "Distance", "Area", "Angle" → Geometry
- "Cycle", "Repeating" → Floyd's cycle detection
- "Power", "Exponent" → Fast exponentiation

**Problem-Solving Framework:**

**1. Look for Mathematical Properties:**
- Can you use a formula instead of simulation?
- Are there symmetries you can exploit?
- Can you reduce the problem using math?

**2. Consider Edge Cases:**
- Zero, negative numbers
- Very large numbers (overflow)
- Empty inputs
- Single element

**3. Optimize with Math:**
- Don't check all divisors, only up to sqrt(n)
- Use fast exponentiation for powers
- Apply modular arithmetic for large numbers

**4. Draw Diagrams:**
- Visualize matrix transformations
- Sketch geometric problems
- Mark boundaries and patterns

**5. Break Down Complex Problems:**
- Matrix rotation = transpose + reverse
- Spiral = process layer by layer
- Complex geometry = break into simpler shapes

**Communication Tips:**
- Explain the mathematical insight
- Walk through small examples
- Discuss alternative approaches
- Mention time/space optimizations
- Show awareness of edge cases

**Common Pitfalls:**
- Integer overflow (especially in combinations/factorials)
- Floating point precision errors
- Off-by-one errors in matrix indices
- Not handling negative numbers properly
- Forgetting to consider in-place constraints`,
      quiz: [
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
            'First: integer overflow (factorial, power grows fast). Second: floating-point precision (compare with epsilon, not ==). Third: modulo errors (forgetting at each step, negative modulo). Fourth: off-by-one in coordinates or loops. Fifth: wrong formula (transpose vs rotation). Sixth: edge cases (0, 1, negative, MAX_INT). Seventh: inefficient algorithm (trial division when Sieve better). My strategy: 1) Use long/BigInteger for large numbers. 2) Compare floats with abs(a-b) < epsilon. 3) Apply modulo at each operation. 4) Draw diagrams for geometry. 5) Double-check formulas. 6) Test: 0, 1, negative, large values. 7) Know complexity of algorithms. For example, factorial: never compute n! then modulo, do modulo at each multiply.',
          keyPoints: [
            'Mistakes: overflow, precision, modulo, off-by-one',
            'Formula errors, edge cases, slow algorithms',
            'Avoid: long types, epsilon, modulo early',
            'Test thoroughly: 0, 1, negative, large',
            'Know: formulas, complexities, alternatives',
            'Think mathematically before coding',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'Matrix rotation in-place: transpose + reverse rows (90° clockwise)',
    'Fast exponentiation reduces O(n) to O(log n) using binary representation',
    'Check divisors only up to sqrt(n) for primality and factorization',
    'Use GCD for simplifying fractions and finding patterns',
    'Spiral/diagonal matrix traversal: track boundaries carefully',
    'Most math problems optimize to O(1) space with clever techniques',
    'Draw diagrams and work through small examples to find patterns',
    'Modular arithmetic prevents overflow and keeps numbers bounded',
  ],
  relatedProblems: ['rotate-image', 'pow-x-n', 'happy-number'],
};
