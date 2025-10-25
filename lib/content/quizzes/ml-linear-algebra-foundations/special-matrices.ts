/**
 * Quiz questions for Special Matrices section
 */

export const specialmatricesQuiz = [
  {
    id: 'special-mat-d1',
    question:
      'Explain why orthogonal matrices preserve lengths and angles. How is this property useful in machine learning, particularly in dimensionality reduction and feature extraction?',
    sampleAnswer:
      'Orthogonal matrices preserve lengths because ||Qv||² = (Qv)ᵀ(Qv) = vᵀQᵀQv = vᵀIv = vᵀv = ||v||². They preserve angles because the dot product is preserved: (Qu)·(Qv) = (Qu)ᵀ(Qv) = uᵀQᵀQv = uᵀv = u·v, and angle is determined by dot product via cos(θ) = (u·v)/(||u||||v||). This property is crucial in ML because it means orthogonal transformations don\'t distort data—they only rotate or reflect it. In PCA, the principal components form an orthonormal basis, so projecting data onto these components preserves relative distances and angles, just in a lower-dimensional space. This ensures no artificial distortion is introduced by the dimensionality reduction. In SVD, the U and V matrices are orthogonal, guaranteeing that the decomposition preserves geometric relationships. In neural networks, some weight initialization schemes use orthogonal matrices to avoid vanishing/exploding gradients. Whitening transformations (decorrelating features) use orthogonal matrices to rotate data into independent components. The key insight: orthogonal transformations are "shape-preserving" transformations that change coordinates without changing the intrinsic geometry of the data, making them ideal for basis changes and feature transformations where we want to maintain data structure.',
    keyPoints: [
      'Orthogonal matrices (QᵀQ=I) preserve lengths and angles: no data distortion',
      'PCA uses orthonormal basis: dimensionality reduction without geometric distortion',
      'Shape-preserving: ideal for basis changes, feature transformations, whitening',
    ],
  },
  {
    id: 'special-mat-d2',
    question:
      'Compare dense versus sparse matrix storage and operations. When should you use each in machine learning, and what are the trade-offs?',
    sampleAnswer:
      "Dense matrices store all n² elements explicitly, while sparse matrices store only non-zero elements (typically many fewer). Trade-offs: Memory: Dense requires O(n²) space, sparse requires O(nnz) where nnz is the number of non-zeros. For a 10,000×10,000 matrix with 0.1% non-zeros, sparse uses 100MB vs dense's 800MB. Speed: For sparse matrices with sparsity s (fraction of zeros), operations are O(nnz) instead of O(n²). However, there's overhead in accessing non-contiguous memory. Operations: Dense supports all operations naturally. Sparse formats (CSR, CSC, COO) optimize different operations—CSR is fast for row operations, CSC for columns, COO for construction. Use sparse when: (1) Data is naturally sparse (>90% zeros): text (document-term matrices), graphs (adjacency matrices), recommender systems (user-item matrices), one-hot encoded features. (2) Scaling to large problems: a million-dimensional sparse vector fits in memory, dense doesn't. Use dense when: (1) Data is dense (<50% zeros): images, audio, embeddings. (2) Need fast, general operations without format constraints. (3) Using GPUs (dense operations more optimized). In practice: Start with dense for simplicity. Switch to sparse if memory/speed becomes an issue. Libraries like scikit-learn support both transparently. Know your data's sparsity: use df.values if dense, use sparse formats for text/graph. Modern deep learning mostly uses dense (images, embeddings) but transformers are exploring sparse attention. The key: sparse is essential for scaling to high-dimensional sparse data (text, graphs), but adds complexity—use only when necessary.",
    keyPoints: [
      'Dense: O(n²) storage, all operations; Sparse: O(nnz) storage, format-dependent ops',
      'Use sparse for >90% zeros (text, graphs, recommenders); dense for images/embeddings',
      'Trade-off: sparse saves memory/computation but adds complexity (CSR/CSC/COO formats)',
    ],
  },
  {
    id: 'special-mat-d3',
    question:
      'Positive definite matrices appear throughout machine learning (covariance matrices, kernels, Hessians at minima). Explain what positive definiteness means intuitively and why it provides useful guarantees in optimization and learning.',
    sampleAnswer:
      'Positive definiteness means xᵀAx > 0 for all non-zero vectors x. Intuitively, it means the quadratic form defined by A is always positive—imagine a bowl shape that curves upward in all directions, never downward or flat (except at the origin). Geometrically, A defines a distance metric that only equals zero at the origin. Mathematical equivalent conditions: all eigenvalues > 0, can be written as A = BᵀB. Why is this important in ML? (1) Covariance matrices: Var (aᵀX) = aᵀΣa where Σ is covariance. Variance must be non-negative, so Σ is positive semi-definite. If features are linearly independent, Σ is positive definite, meaning the distribution is non-degenerate (has spread in all principal directions). (2) Optimization: The Hessian at a local minimum must be positive definite—it curves upward in all directions. This guarantees the minimum is strict and unique (locally). In convex optimization, positive definite Hessian everywhere means globally convex. (3) Kernels: Valid kernel functions produce positive definite kernel matrices K, which guarantees a unique solution in kernel methods like SVMs and ensures the optimization problem is convex. (4) Numerical stability: Positive definite matrices are invertible (no zero eigenvalues) and well-conditioned (eigenvalues bounded away from zero), making numerical algorithms stable. (5) Cholesky decomposition: Only positive definite matrices have Cholesky decomposition A = LLᵀ, a fast and stable way to solve systems and generate multivariate Gaussian samples. In summary: positive definiteness provides convexity guarantees (unique solutions), stability guarantees (invertible, well-conditioned), and enables efficient algorithms (Cholesky). Recognizing when matrices are positive definite allows using specialized, faster, and more stable algorithms.',
    keyPoints: [
      'Positive definite: xᵀAx > 0 (bowl-shaped), all eigenvalues > 0, invertible',
      'Guarantees: covariance non-degenerate, Hessian at minima convex, kernels valid',
      'Enables: Cholesky decomposition, numerical stability, unique solutions',
    ],
  },
];
