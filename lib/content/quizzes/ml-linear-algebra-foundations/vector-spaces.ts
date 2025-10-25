/**
 * Quiz questions for Vector Spaces section
 */

export const vectorspacesQuiz = [
  {
    id: 'vec-space-d1',
    question:
      'Explain why a line through the origin is a subspace of ℝ², but a line not through the origin is not. Use both algebraic (axioms) and geometric arguments.',
    sampleAnswer:
      'A line through the origin in ℝ² has the form {t·v : t ∈ ℝ} for some direction vector v. Algebraically, this satisfies subspace requirements: (1) Contains zero: t=0 gives 0·v = 0 ✓ (2) Closed under addition: t₁v + t₂v = (t₁+t₂)v, still on the line ✓ (3) Closed under scalar multiplication: c (tv) = (ct)v, still on the line ✓ Geometrically: adding two vectors on the line through origin gives another vector on the same line (parallelogram law), and scaling a vector keeps it on the same line. A line NOT through origin, say y = 2x + 1, fails multiple requirements: (1) No zero vector: (0,0) is not on the line since 0 ≠ 2·0 + 1 ✗ (2) Not closed under addition: points (0,1) and (1,3) are both on the line, but their sum (1,4) is not (4 ≠ 2·1 + 1) ✗ Geometrically: adding two position vectors on a shifted line produces a vector that "jumps" away from the line. The geometric intuition is that subspaces must include the origin (natural reference point for vector addition) and must contain all scaled versions and sums of their vectors. A shifted line is missing the origin, so vector operations escape the line. In ML context: centering data (subtracting mean) transforms our data cloud to pass through the origin, converting it into a proper subspace where linear operations behave nicely.',
    keyPoints: [
      'Subspace requirements: contains zero, closed under addition and scalar multiplication',
      'Line through origin: satisfies all axioms; shifted line: fails (no zero, not closed)',
      'ML: centering data (subtract mean) makes it pass through origin (proper subspace)',
    ],
  },
  {
    id: 'vec-space-d2',
    question:
      'The rank-nullity theorem states that for an m×n matrix A: rank(A) + dim (null(A)) = n. Explain this theorem intuitively and discuss its significance in understanding linear transformations and solving Ax = b.',
    sampleAnswer:
      'The rank-nullity theorem reveals a fundamental trade-off: the n input dimensions are partitioned into two complementary spaces. Rank(A) = dimension of column space = number of independent output dimensions that A can produce. Dim (null(A)) = nullity = number of independent input directions that get mapped to zero. Together they must sum to n (total input dimensions). Intuition: imagine A as a transformation. The null space consists of inputs that A "destroys" (maps to zero). The rank counts how many independent directions survive the transformation. Every input dimension either contributes to output (counted in rank) or gets destroyed (counted in nullity). For solving Ax = b: (1) If b is in column space (possible with rank dimensions), solutions exist. (2) If nullity > 0, there are multiple solutions—the null space provides "free parameters" that can be added without changing Ax. The solution set is x_particular + null(A), an affine subspace of dimension = nullity. (3) If nullity = 0 (full column rank), solutions are unique when they exist. Example: A is 3×5 with rank 3. Then nullity = 5-3 = 2. This means: (a) A can produce any vector in a 3D subspace of ℝ³ (column space). (b) For any b in that subspace, there are infinitely many solutions forming a 2D affine subspace (particular solution + 2D null space). In ML: underdetermined systems (more unknowns than equations) always have nullity > 0, giving infinitely many solutions. We typically choose the minimum norm solution (closest to origin). Overdetermined systems typically have nullity = 0, and we use least squares. Understanding the rank-nullity theorem helps diagnose whether a system has no solution, unique solution, or infinite solutions, and explains why regularization (adding constraints) is needed for underdetermined problems.',
    keyPoints: [
      'rank(A) + nullity(A) = n: input dimensions partition into output and destroyed',
      'Nullity > 0: infinite solutions (null space = free parameters); nullity = 0: unique',
      'ML: underdetermined has nullity > 0, use minimum norm solution or regularization',
    ],
  },
  {
    id: 'vec-space-d3',
    question:
      'In machine learning, feature matrices with linearly dependent columns can cause problems. Explain what linear dependence means geometrically, why it causes issues computationally and statistically, and how techniques like PCA address this.',
    sampleAnswer:
      "Linear dependence means some features are redundant—they can be written as combinations of other features, adding no new information. Geometrically: if features f₁, f₂, f₃ are linearly dependent, they don't span a 3D space but only a 2D plane (or even 1D line). Data points lie in a lower-dimensional subspace than the ambient feature space suggests. For example, if f₃ = 2f₁ + f₂ always, the data lies on a 2D plane in 3D space. Computational issues: (1) Near-singular matrices: XᵀX becomes nearly singular (determinant ≈ 0), making (XᵀX)⁻¹ numerically unstable or undefined. Small errors get amplified. (2) Non-unique solutions: In regression, w = (XᵀX)⁻¹Xᵀy fails if XᵀX is singular. Multiple weight combinations produce identical predictions—the problem is underdetermined. (3) Inflated coefficients: With multicollinearity (high but not perfect correlation), coefficients become unstable and interpretation breaks down. Small data changes cause wild coefficient swings. Statistical issues: (1) Variance inflation: Standard errors of coefficients explode, making hypothesis testing unreliable. (2) Loss of interpretability: Can't isolate individual feature effects when they're entangled. (3) Overfitting: Unnecessary parameters waste degrees of freedom. PCA addresses this by: (1) Finding orthogonal (linearly independent) principal components, removing redundancy. (2) Ordered by variance: first PC captures most variation, last PCs capture noise/redundancy. (3) Dimensionality reduction: keeping top k PCs gives k truly independent features spanning the same subspace as original data. This regularizes the problem, improving numerical stability and generalization. Alternative solutions: (1) Feature selection: manually remove redundant features. (2) Ridge regression: add λI to XᵀX, making it invertible even if singular. (3) Feature engineering: create genuinely independent features. Understanding linear independence helps diagnose multicollinearity (check rank or condition number), explains why regularization helps, and motivates dimensionality reduction techniques.",
    keyPoints: [
      'Linear dependence: redundant features, data in lower-dimensional subspace',
      'Issues: singular XᵀX, non-unique solutions, inflated coefficients, poor generalization',
      'Solutions: PCA (orthogonal components), Ridge (regularize), feature selection',
    ],
  },
];
