/**
 * Quiz questions for Matrix Operations section
 */

export const matrixoperationsQuiz = [
  {
    id: 'mat-ops-d1',
    question:
      'Explain how NumPy broadcasting works and why it is useful in machine learning. Provide examples of common broadcasting patterns in neural networks.',
    sampleAnswer:
      "Broadcasting is NumPy\'s mechanism for performing operations on arrays of different shapes by automatically expanding smaller dimensions. Rules: (1) If arrays differ in number of dimensions, prepend 1s to the smaller shape. (2) Arrays are compatible if each dimension is either equal or one is 1. (3) Arrays with dimension 1 are stretched to match the other dimension. This is invaluable in ML: (1) Adding bias: In a layer output = XW + b, if X@W has shape (batch, features) and b has shape (features,), broadcasting adds b to every sample automatically. Without broadcasting, we'd need explicit loops or tile b to (batch, features). (2) Batch normalization: (X - mean) / std where mean and std are per-feature statistics broadcasts automatically. (3) Attention mechanisms: When computing attention scores, we often need to broadcast queries/keys across different dimensions. (4) Loss computation: Comparing predictions (batch, classes) with one-hot labels involves broadcasting. Broadcasting makes code concise, readable, and fast (vectorized operations instead of loops). It eliminates the need for manually expanding arrays, reducing memory and computation. Understanding broadcasting is essential for implementing efficient ML code and debugging shape errors.",
    keyPoints: [
      'Broadcasting: automatic array expansion when dimensions are compatible (equal or 1)',
      'ML use cases: bias addition, batch norm, attention mechanisms (eliminates loops)',
      'Makes code concise, readable, fast; essential for efficient ML implementation',
    ],
  },
  {
    id: 'mat-ops-d2',
    question:
      'Compare the three perspectives of matrix multiplication: (1) element-wise computation, (2) column combination, (3) row transformation. When is each perspective most useful in understanding ML operations?',
    sampleAnswer:
      "Matrix multiplication C = AB can be understood three ways, each illuminating different aspects: (1) Element-wise: cᵢⱼ = Σₖ aᵢₖbₖⱼ. This is how we compute it manually and verify calculations. Useful for understanding computational complexity (O(n³) for n×n matrices) and debugging specific element calculations. (2) Column perspective: AB = [Ab₁ | Ab₂ | ... | Abₙ], each column of C is A transforming the corresponding column of B. This perspective is crucial for understanding how weight matrices transform input feature dimensions in neural networks. When we compute Y = XW, each output feature (column of Y) is a specific learned combination of input features. (3) Row perspective: Each row of C is a row of A combining rows of B. Useful for batch processing understanding: when X is (batch, features) and W is (features, outputs), each row of Y represents one sample's transformation. In backpropagation, the row perspective helps understand how gradients flow through layers. The column view is best for understanding feature transformations and weight matrix interpretation. The row view is best for understanding batch processing and sample-wise operations. The element view is best for mathematical verification and complexity analysis. Expert ML practitioners fluidly switch between perspectives depending on whether they are analyzing feature engineering, batch processing, or debugging.",
    keyPoints: [
      'Element-wise (Σₖ aᵢₖbₖⱼ): computational complexity, manual verification',
      'Column perspective: feature transformations, how weights combine input features',
      'Row perspective: batch processing, sample-wise transformations, gradient flow',
    ],
  },
  {
    id: 'mat-ops-d3',
    question:
      'The trace operation has the property tr(AB) = tr(BA) even though AB ≠ BA. Explain why this is true and discuss where this property is useful in machine learning and statistics.',
    sampleAnswer:
      'The trace property tr(AB) = tr(BA) is remarkable because matrix multiplication is not commutative. Proof sketch: tr(AB) = Σᵢ(AB)ᵢᵢ = Σᵢ Σₖ aᵢₖbₖᵢ. Similarly, tr(BA) = Σₖ(BA)ₖₖ = Σₖ Σᵢ bₖᵢaᵢₖ. These are the same sum with indices reversed. Geometrically, trace is invariant to cyclic permutations: tr(ABC) = tr(CAB) = tr(BCA). This property is crucial in several ML contexts: (1) Matrix derivatives: When deriving gradients, we often need to simplify expressions like tr(X^T A X). The cyclic property allows us to rearrange terms. (2) Frobenius norm: ||A||_F² = tr(A^T A) = tr(AA^T), giving equivalent expressions. (3) PCA and covariance: tr(Σ) gives total variance, and trace properties help simplify eigenvalue computations. (4) Fisher Information Matrix: In statistics, trace properties simplify expected value calculations. (5) Loss functions: Some regularization terms use trace, and the cyclic property helps derive gradients. (6) Quantum mechanics and physics: Trace is basis-independent, making it fundamental for observable quantities. The trace is one of the few matrix operations that treats AB and BA equivalently, making it especially useful when we need quantities invariant to certain transformations. In ML optimization, recognizing when to use trace properties can simplify complex derivative calculations significantly.',
    keyPoints: [
      'tr(AB) = tr(BA): cyclic permutation invariance despite AB ≠ BA',
      'Proof: both equal Σᵢ Σₖ aᵢₖbₖᵢ (same sum, reordered indices)',
      'ML uses: matrix derivatives, Frobenius norm, covariance, regularization terms',
    ],
  },
];
