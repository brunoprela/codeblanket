export const linearAlgebraProblemsQuiz = [
  {
    id: 'lap-q-1',
    question:
      'Jane Street: "Given a 3-asset portfolio with covariance matrix Σ and weight vector w = [0.5, 0.3, 0.2]ᵀ, explain step-by-step how to compute portfolio variance σ_p² = wᵀΣw. Then, if Σ has eigenvalues [4, 2, 1], what does this tell you about risk concentration? Finally, how would you find the minimum variance portfolio without calculus using linear algebra?" Show complete matrix operations and risk interpretation.',
    sampleAnswer:
      'Complete portfolio variance calculation: (1) Computing wᵀΣw: This is a quadratic form. Step 1: Multiply Σw (matrix-vector product) to get a 3×1 vector v. Step 2: Multiply wᵀv (dot product) to get scalar variance. Example with specific Σ: Σ = [σ₁² ρ₁₂σ₁σ₂ ρ₁₃σ₁σ₃; ρ₁₂σ₁σ₂ σ₂² ρ₂₃σ₂σ₃; ρ₁₃σ₁σ₃ ρ₂₃σ₂σ₃ σ₃²]. First, v = Σw computes weighted covariances: v_i = Σⱼ Σᵢⱼwⱼ. Then σ_p² = Σᵢ wᵢvᵢ. (2) Eigenvalue interpretation: Eigenvalues [4, 2, 1] with sum = 7 represent variance along principal components. λ₁=4 explains 4/7≈57% of total variance (most important risk factor). λ₂=2 explains 29%, λ₃=1 explains 14%. Risk is moderately concentrated in first PC (not too concentrated would be like [6.5, 0.3, 0.2]; very concentrated would be [6.9, 0.05, 0.05]). Diversification benefit: if risks were independent, total would be sum of individual variances. Eigenvalues show correlated risk structure. (3) Minimum variance portfolio without calculus: Goal: find w minimizing wᵀΣw subject to wᵀ1=1 (weights sum to 1). Linear algebra solution: w_min = Σ⁻¹1 / (1ᵀΣ⁻¹1), where 1 = [1,1,1]ᵀ. Derivation: use Lagrangian L = wᵀΣw - λ(wᵀ1-1). First-order condition: ∂L/∂w = 2Σw - λ1 = 0 → Σw = (λ/2)1. Multiply both sides by Σ⁻¹: w = (λ/2)Σ⁻¹1. Apply constraint wᵀ1 = 1: (λ/2)(1ᵀΣ⁻¹1) = 1 → λ/2 = 1/(1ᵀΣ⁻¹1). Therefore: w_min = Σ⁻¹1 / (1ᵀΣ⁻¹1). This is the global minimum variance portfolio. To compute: (a) invert Σ to get Σ⁻¹, (b) multiply Σ⁻¹ by 1 vector, (c) normalize by sum. Interview communication: explain matrix multiplication order, interpret eigenvalues as risk factors, show closed-form solution for minimum variance using Lagrange multipliers and linear algebra.',
    keyPoints: [
      'Portfolio variance: σ_p² = wᵀΣw, compute as w·(Σw)',
      'Eigenvalues [4,2,1] sum to 7, largest explains 57% of variance',
      'Risk concentration: λ₁/Σλᵢ measures fraction in first factor',
      'Min variance portfolio: w = Σ⁻¹1 / (1ᵀΣ⁻¹1) from Lagrangian',
      'Larger eigenvalue ratio indicates more concentrated risk',
    ],
  },
  {
    id: 'lap-q-2',
    question:
      'Citadel: "You have correlation matrix ρ for 4 assets. You observe one eigenvalue is negative. What does this mean? Is this possible for a true correlation matrix? How would you fix this matrix to make it valid? Then, given valid correlation matrix with eigenvalues [2.5, 0.8, 0.5, 0.2], explain the risk structure and compute the condition number. What does condition number tell you about numerical stability?" Provide complete linear algebra analysis.',
    sampleAnswer:
      'Complete correlation matrix analysis: (1) Negative eigenvalue interpretation: A negative eigenvalue violates positive semi-definiteness. For correlation matrix ρ, we must have xᵀρx ≥ 0 for all x (variance cannot be negative). If λ < 0, there exists eigenvector v such that vᵀρv = λ||v||² < 0, which is impossible for valid correlation. Conclusion: matrix is NOT a valid correlation matrix. (2) How this happens: Estimated correlation matrices from small samples can have numerical errors leading to negative eigenvalues. Incompatible pairwise correlations (e.g., ρ₁₂=0.9, ρ₂₃=0.9, ρ₁₃=-0.9) create mathematical inconsistency. (3) Fixing the matrix: Method A (eigenvalue flooring): compute eigendecomposition ρ = QΛQᵀ, set negative eigenvalues to small positive (e.g., 0.001), reconstruct ρ_fixed = QΛ_fixedQᵀ, rescale diagonal to 1s. Method B (shrinkage): ρ_fixed = αρ + (1-α)I where α<1, moves toward identity matrix. Method C (nearest positive semidefinite): solve optimization min||ρ_fixed - ρ||_F subject to ρ_fixed ⪰ 0 (semidefinite programming). (4) Eigenvalue analysis [2.5, 0.8, 0.5, 0.2]: Sum = 4 (correct for 4×4 correlation matrix - trace equals dimension). First PC explains 2.5/4 = 62.5% of variance (strong common factor). Remaining 37.5% split among 3 factors (20%, 12.5%, 5%). Risk structure: highly correlated assets with one dominant systematic risk factor. (5) Condition number: κ(ρ) = λ_max/λ_min = 2.5/0.2 = 12.5. Interpretation: matrix is moderately ill-conditioned. κ=1 is perfectly conditioned (identity). κ>10 indicates some numerical instability. κ>100 is severely ill-conditioned. For κ=12.5: matrix inversion is reliable but with some sensitivity to perturbations. In portfolio optimization, small changes in correlations could lead to noticeable changes in optimal weights. Practical implication: use regularization (shrinkage) to improve conditioning. Interview: demonstrate understanding of positive definiteness, explain why negative eigenvalues are invalid, show multiple fixing methods, interpret condition number for numerical stability.',
    keyPoints: [
      'Negative eigenvalue violates positive definiteness (impossible for valid correlation)',
      'Fix via: eigenvalue flooring, shrinkage toward identity, or optimization',
      'Eigenvalues [2.5,0.8,0.5,0.2] sum to 4, first explains 62.5% variance',
      'Condition number κ = 2.5/0.2 = 12.5 (moderately conditioned)',
      'High κ indicates numerical instability in matrix inversion',
    ],
  },
  {
    id: 'lap-q-3',
    question:
      'Two Sigma: "Prove that for any square matrix A, the matrix AᵀA is positive semi-definite. Then, explain why covariance matrices are always positive semi-definite using this result. Finally, if you have data matrix X (n observations × p features) and compute sample covariance S = (1/n)XᵀX, under what conditions is S singular (non-invertible)? Relate this to the n vs p problem in statistics." Provide rigorous mathematical proof and practical implications.',
    sampleAnswer:
      "Complete proof and analysis: (1) Proof that AᵀA is positive semi-definite: For any vector x, we need to show xᵀ(AᵀA)x ≥ 0. Proof: xᵀ(AᵀA)x = (xᵀAᵀ)(Ax) = (Ax)ᵀ(Ax) = ||Ax||² ≥ 0. Since norm squared is always non-negative, AᵀA is positive semi-definite. Furthermore, xᵀ(AᵀA)x = 0 iff Ax = 0 (equality only when x is in null space of A). Therefore, AᵀA is positive definite iff A has full column rank (no null space except 0). (2) Covariance matrix application: Let X be n×p data matrix (mean-centered). Covariance matrix Σ = (1/n)XᵀX. By result above, Σ is positive semi-definite. Physical interpretation: for any portfolio weight vector w, variance wᵀΣw = (1/n)||Xw||² ≥ 0 (variance cannot be negative). This explains why covariance matrices must be PSD - they measure variances. (3) Singularity conditions for S = (1/n)XᵀX: S is singular (det(S)=0) iff AᵀA has eigenvalue 0 iff A doesn't have full column rank. For n×p matrix X: rank(X) ≤ min (n,p). Case 1: If p > n (more features than observations): rank(X) ≤ n < p → XᵀX is at most rank n → S is singular (at least p-n zero eigenvalues). Case 2: If p ≤ n but features are linearly dependent: rank(X) < p → S is singular. Case 3: If p ≤ n and features are linearly independent: rank(X) = p → S is invertible (positive definite). (4) The n vs p problem: In modern statistics (genomics, finance with many assets): p >> n is common. Example: p=100 stocks, n=60 days of data. Sample covariance S is 100×100 but rank ≤ 60 → singular → cannot invert for portfolio optimization. Solutions: (a) Regularization: use shrinkage S_shrink = αS + (1-α)I, (b) Factor models: reduce dimension via PCA, (c) Increase n: get more data, (d) Reduce p: feature selection. Practical implication: cannot use classical Markowitz optimization with p > n without regularization. Interview: prove PSD using ||Ax||², connect to covariance interpretation, explain rank deficiency when p>n, discuss practical solutions for high-dimensional problems.",
    keyPoints: [
      'Proof: xᵀ(AᵀA)x = ||Ax||² ≥ 0, so AᵀA is always PSD',
      'Covariance Σ = (1/n)XᵀX inherits PSD property from structure',
      'Singular when p > n (more features than samples) → rank deficient',
      'In p > n regime: S has at most n non-zero eigenvalues',
      'Solutions: regularization, factor models, dimensionality reduction',
    ],
  },
];
