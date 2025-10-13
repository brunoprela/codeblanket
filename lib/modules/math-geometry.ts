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
