/**
 * Quiz questions for Systems of Linear Equations section
 */

export const systemslinearequationsQuiz = [
  {
    id: 'sys-lin-d1',
    question:
      'Linear regression can be solved using normal equations (AᵀAx = Aᵀb) or QR decomposition. Compare these approaches in terms of computational cost, numerical stability, and when each should be used.',
    sampleAnswer:
      'Normal equations involve computing AᵀA and solving (AᵀA)x = Aᵀb. Computational cost: Computing AᵀA is O(mn²) for A with shape (m,n), then solving is O(n³), total O(mn² + n³). For m >> n, this is approximately O(mn²). QR decomposition computes A = QR in O(mn²), then solves Rx = Qᵀb in O(n²), total O(mn²). Both have similar asymptotic complexity. However, numerical stability differs dramatically. Normal equations square the condition number: κ(AᵀA) = κ(A)². If A has condition number 10⁴, AᵀA has condition number 10⁸, causing severe numerical errors. QR maintains the original condition number κ(R) ≈ κ(A), providing much better stability. Use normal equations when: (1) A is very well-conditioned (κ < 100), (2) AᵀA is already computed for other reasons, (3) Memory is severely constrained. Use QR when: (1) A is ill-conditioned, (2) Numerical accuracy is critical, (3) Standard case (recommended default). In practice: sklearn uses QR-based solvers by default. Normal equations are mainly of historical/educational interest. Modern recommendation: always use QR or SVD-based methods unless you have specific constraints.',
    keyPoints: [
      'Both O(mn²) complexity, but QR numerically superior',
      'Normal equations square condition number: κ(AᵀA) = κ(A)² (unstable)',
      'QR recommended default (sklearn uses it); normal equations only if well-conditioned',
    ],
  },
  {
    id: 'sys-lin-d2',
    question:
      'Explain why solving Ax = b by computing x = A⁻¹b is considered bad practice compared to using specialized solvers like np.linalg.solve(). What are the computational and numerical reasons?',
    sampleAnswer:
      "Computing x = A⁻¹b is bad practice for three main reasons: (1) Computational cost: Computing A⁻¹ explicitly costs O(n³). Then multiplying A⁻¹b costs O(n²), total O(n³). Using LU decomposition: PA = LU costs O(n³), then forward/back substitution costs O(n²), total O(n³). While asymptotically equal, LU has a smaller constant factor and avoids storing the full inverse matrix (n² space vs n space for the factorization). (2) Numerical stability: Matrix inversion accumulates rounding errors. The inverse A⁻¹ may have large entries even if A is well-behaved, amplifying errors. Specialized solvers use techniques like partial pivoting, scaling, and iterative refinement to minimize error propagation. For ill-conditioned systems, the difference can be several orders of magnitude in accuracy. (3) Flexibility: Computing A⁻¹ doesn't work for rectangular matrices or singular systems. LU/QR-based solvers can handle these cases (using least squares or finding minimum norm solutions). They also provide diagnostics like rank, condition number estimates, and residual norms. In practice: np.linalg.solve() uses LAPACK's optimized routines with partial pivoting and is typically 2-3x faster than computing the inverse, plus more accurate. The only time to compute A⁻¹ explicitly is when you actually need the inverse matrix itself (e.g., computing covariance inverse for Mahalanobis distance, or theoretical analysis). Even then, specialized methods often exist (e.g., Cholesky for positive definite matrices).",
    keyPoints: [
      'Computing A⁻¹ explicitly: same O(n³) cost, but worse numerical stability',
      'np.linalg.solve(): 2-3x faster, more accurate, better error handling',
      'Only compute A⁻¹ when you need the full inverse matrix (rare)',
    ],
  },
  {
    id: 'sys-lin-d3',
    question:
      'In machine learning, we often encounter regularized least squares (Ridge regression): minimize ||Ax - b||² + λ||x||². Explain how this modifies the normal equations and why regularization helps with ill-conditioned or underdetermined systems.',
    sampleAnswer:
      "Regularized least squares adds a penalty term λ||x||² to prevent overfitting. Taking the gradient and setting to zero: ∇(||Ax-b||² + λ||x||²) = 2Aᵀ(Ax-b) + 2λx = 0. This gives modified normal equations: (AᵀA + λI)x = Aᵀb. Compare to unregularized: AᵀAx = Aᵀb. The addition of λI (ridge term) has profound effects: (1) For ill-conditioned systems: AᵀA might have condition number κ = 10¹⁰. Adding λI increases all eigenvalues by λ, improving conditioning. If λ = 0.01 and smallest eigenvalue is 10⁻⁸, it becomes 10⁻² + 10⁻⁸ ≈ 10⁻², dramatically reducing κ. (2) For underdetermined systems (m < n): AᵀA is rank-deficient (not invertible). Adding λI makes (AᵀA + λI) full rank and invertible, providing a unique solution. (3) For nearly collinear features: High correlation creates near-zero eigenvalues in AᵀA. Regularization prevents coefficients from exploding to compensate for near-singularity. Geometric interpretation: λI shrinks all coefficients toward zero, preferring simpler models. This trades bias (slightly worse training fit) for variance (much better generalization). The optimal λ balances underfitting vs overfitting, typically chosen by cross-validation. Computational benefit: (AᵀA + λI) is better conditioned than AᵀA, so solving is more stable. For large λ, it's nearly diagonal, making solution very stable. In practice: Ridge regression is standard for high-dimensional problems (n large, possible multicollinearity). It's equivalent to imposing a Gaussian prior on weights in Bayesian framework: x ~ N(0, (1/λ)I). Modern variants include elastic net (L1 + L2 penalties) and adaptive regularization (different λ per feature).",
    keyPoints: [
      'Ridge modifies normal equations: (AᵀA + λI)x = Aᵀb (adds λI term)',
      'λI improves conditioning (increases eigenvalues), makes underdetermined solvable',
      'Trades bias for variance: shrinks coefficients, better generalization',
    ],
  },
];
