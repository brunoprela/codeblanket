/**
 * Quiz questions for Vector Operations section
 */

export const vectoroperationsQuiz = [
  {
    id: 'vec-ops-d1',
    question:
      'Compare and contrast Euclidean distance and cosine similarity. When would you choose one over the other in a machine learning application? Provide specific examples.',
    sampleAnswer:
      'Euclidean distance measures absolute spatial distance between vectors, while cosine similarity measures the angle between vectors, ignoring magnitude. Euclidean distance is sensitive to both direction and magnitude: vectors [1,1] and [10,10] are far apart (distance ≈ 12.7) even though they point in the same direction. Cosine similarity cares only about direction: these vectors have similarity = 1.0 (identical direction). Choose Euclidean when magnitude matters: measuring physical distances, detecting anomalies in sensor data, or comparing feature vectors where scale is meaningful. Choose cosine when direction matters more than magnitude: text similarity (a long document and short document can be similar if they discuss the same topics), recommendation systems (user preference patterns matter more than absolute ratings), and word embeddings (semantic similarity). In practice, for high-dimensional sparse data like text, cosine similarity often performs better because it is less affected by varying document lengths and focuses on content distribution.',
    keyPoints: [
      'Euclidean distance: sensitive to magnitude and direction (physical distance)',
      'Cosine similarity: only direction matters, ignores magnitude (text/embeddings)',
      'Choose Euclidean when scale matters, cosine when pattern/direction matters',
    ],
  },
  {
    id: 'vec-ops-d2',
    question:
      'Explain why L1 regularization (Lasso) tends to produce sparse models while L2 regularization (Ridge) does not. How do the different properties of L1 and L2 norms lead to this behavior?',
    sampleAnswer:
      'L1 and L2 regularization penalize model complexity differently due to the geometry of their norms. L2 regularization adds λ||w||₂² = λ(w₁² + w₂² + ... + wₙ²) to the loss, which creates a smooth, differentiable penalty. Gradients of L2 are proportional to the weights themselves, so weights shrink proportionally but rarely reach exactly zero. Geometrically, L2 creates a circular constraint region—the optimal solution typically lies where the elliptical error contours touch the circle, which is usually not at an axis (where weights are zero). L1 regularization adds λ||w||₁ = λ(|w₁| + |w₂| + ... + |wₙ|), creating a diamond-shaped constraint region with sharp corners at the axes. The penalty is constant regardless of weight magnitude (gradient is ±1), so small weights are penalized as much as large weights, driving them to exactly zero. The corners of the L1 diamond align with coordinate axes, so solutions often occur at corners where many weights are zero, producing sparsity. This makes L1 ideal for feature selection: it automatically identifies and zeros out irrelevant features, while L2 keeps all features but shrinks them.',
    keyPoints: [
      'L1 diamond geometry has corners on axes where weights = 0 (sparsity)',
      'L2 circular geometry rarely touches axes, shrinks weights proportionally',
      'L1 constant gradient drives small weights to zero; L2 gradient ∝ weight',
    ],
  },
  {
    id: 'vec-ops-d3',
    question:
      'The dot product can be computed as a sum of element-wise products or as ||u|| ||v|| cos(θ). Discuss how these two perspectives are used differently in machine learning, providing examples of each.',
    sampleAnswer:
      "These two equivalent definitions of the dot product serve different conceptual purposes in ML. The algebraic definition (u · v = Σ uᵢvᵢ) is used for **computation**: it is how we actually calculate dot products efficiently in code, and it is how we think about the mechanics of neural networks. Each neuron computes Σ wᵢxᵢ + b—a weighted sum of inputs. This perspective emphasizes the dot product as an aggregation operation, combining information from multiple sources with learned weights. It is also how we compute distances (||u - v||² = (u-v) · (u-v)) and norms (||v||² = v · v). The geometric definition (u · v = ||u|| ||v|| cos(θ)) is used for **interpretation** and **analysis**: it reveals that the dot product measures alignment or similarity between vectors. Cosine similarity for text comparison uses this explicitly. In neural networks, we can interpret what a neuron is computing: it is measuring how aligned the input is with the neuron's weight vector. Large positive dot product means strong alignment (similar direction), negative means opposite directions, zero means orthogonal (independent). This geometric view also explains why normalized vectors are important: after normalization, the dot product is purely the cosine of the angle, isolating directional similarity from magnitude effects. Both perspectives are essential—algebraic for implementation, geometric for understanding.",
    keyPoints: [
      'Algebraic (Σ uᵢvᵢ): used for computation in neural networks and algorithms',
      'Geometric (||u||||v||cos θ): used for interpretation of similarity/alignment',
      'Both perspectives essential: algebraic for implementation, geometric for understanding',
    ],
  },
];
