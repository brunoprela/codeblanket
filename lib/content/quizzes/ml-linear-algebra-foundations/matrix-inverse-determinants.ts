/**
 * Quiz questions for Matrix Inverse & Determinants section
 */

export const matrixinversedeterminantsQuiz = [
  {
    id: 'inv-det-d1',
    question:
      'Explain the geometric interpretation of a singular matrix (determinant = 0). What happens to space under this transformation, and why does this mean the matrix has no inverse?',
    sampleAnswer:
      "A singular matrix (det = 0) collapses space to a lower dimension. In 2D, it might collapse the plane to a line or point. In 3D, it might collapse 3D space to a plane, line, or point. Geometrically, imagine a transformation that squashes a square into a line—the area becomes zero, hence determinant = 0. This transformation loses information: multiple points map to the same output point (e.g., the entire square becomes a single line). An inverse would need to \"un-collapse\" this—to take a line back to a square—but this is impossible because we've lost the information about which point on the line came from which point in the square. Mathematically, the columns of a singular matrix are linearly dependent: one column is a combination of others. This means the transformation doesn't span the full output space—it only reaches a lower-dimensional subspace. Since the transformation isn't onto (doesn't cover all output space), it can't be inverted. In ML context: if your data matrix is singular, features are redundant (linearly dependent), and you can't solve certain systems uniquely. This is why we check rank and condition number—nearly singular matrices cause numerical instability even if not exactly singular.",
    keyPoints: [
      'det=0: space collapses to lower dimension, transformation loses information',
      'Columns linearly dependent: transformation not onto full output space',
      "ML impact: redundant features, can't solve systems uniquely, numerical instability",
    ],
  },
  {
    id: 'inv-det-d2',
    question:
      'The property (AB)⁻¹ = B⁻¹A⁻¹ shows that order reverses when inverting a product. Explain why this must be true using the concept of transformations "undoing" each other.',
    sampleAnswer:
      "Think of matrix multiplication as composing transformations applied right to left: AB means \"first apply B, then apply A.\" To undo this composition, we must undo the operations in reverse order: first undo A, then undo B. This is like putting on socks then shoes—to reverse it, you remove shoes first (undo A), then socks (undo B). Formally: suppose we want to verify (AB)(B⁻¹A⁻¹) = I. Expanding: (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I. The middle BB⁻¹ = I cancels out, leaving AA⁻¹ = I. If we tried (AB)(A⁻¹B⁻¹) instead, we'd get (AB)(A⁻¹B⁻¹) = A(BA⁻¹)B⁻¹, and BA⁻¹ ≠ I in general (matrices don't commute), so this doesn't work. In ML: this appears in backpropagation through composed layers. If forward pass is y = Layer2(Layer1(x)), the backward pass must go through Layer2's gradient first, then Layer1's—the reverse order. Understanding this reversal is crucial for implementing custom neural network layers and deriving gradients correctly. It also explains why (ABC)⁻¹ = C⁻¹B⁻¹A⁻¹—complete reversal for any number of matrices.",
    keyPoints: [
      '(AB)⁻¹ = B⁻¹A⁻¹: undo operations in reverse order (like socks then shoes)',
      'Proof: (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I',
      'ML: backpropagation through layers undoes forward pass in reverse order',
    ],
  },
  {
    id: 'inv-det-d3',
    question:
      'Discuss the condition number of a matrix and its importance in machine learning. What problems arise with ill-conditioned matrices, and how can you detect and mitigate them?',
    sampleAnswer:
      'The condition number κ(A) = ||A|| ||A⁻¹|| measures sensitivity to numerical errors. Small condition number (close to 1): well-conditioned, stable. Large condition number: ill-conditioned, small input changes or rounding errors cause large output changes. Why it matters: Computers use finite precision (typically 64-bit floats with ~15 significant digits). With κ(A) ≈ 10^k, you lose about k digits of precision. If κ ≈ 10^10, you lose 10 digits, leaving only ~5 accurate digits. This manifests as: (1) Solving Ax=b becomes inaccurate, (2) Gradient computations in optimization become unreliable, (3) Eigenvalue/SVD calculations may be wrong. Common causes in ML: (1) Features with vastly different scales (e.g., age in years vs income in dollars—scale difference of ~1000x), (2) Highly correlated features (multicollinearity), (3) Near-duplicate rows/columns in data matrix, (4) Using raw polynomials without orthogonalization. Detection: Check np.linalg.cond(A). Rule of thumb: κ > 10^10 is problematic, κ > 10^15 is critical. Mitigation: (1) Feature scaling/standardization: Scale all features to similar ranges (StandardScaler), (2) Regularization: Add λI to covariance matrix (Ridge regression), (3) PCA: Remove correlated dimensions, (4) Higher precision: Use float128 if needed, (5) Specialized algorithms: Use QR decomposition instead of normal equations for least squares. In deep learning: proper weight initialization and batch normalization help keep condition numbers reasonable throughout training. Understanding condition numbers helps diagnose why a model might be numerically unstable or producing nonsensical results despite correct code.',
    keyPoints: [
      'κ(A) = ||A|| ||A⁻¹||: measures numerical stability (κ ≈ 10^k loses k digits precision)',
      'Causes: vastly different feature scales, correlated features, multicollinearity',
      'Mitigation: feature scaling, regularization (Ridge), PCA, QR decomposition',
    ],
  },
];
