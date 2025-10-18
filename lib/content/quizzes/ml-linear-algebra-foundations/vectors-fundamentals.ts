/**
 * Quiz questions for Vectors Fundamentals section
 */

export const vectorsfundamentalsQuiz = [
  {
    id: 'vec-fund-d1',
    question:
      'Explain why normalizing feature vectors is important in machine learning algorithms like k-NN and neural networks. What problems can occur if features are not normalized?',
    sampleAnswer:
      'Feature normalization is crucial because it ensures all features contribute equally to distance calculations and gradient updates. Without normalization, features with larger scales dominate. For example, in k-NN, if one feature is "house price in dollars" (range: 100,000-500,000) and another is "number of bedrooms" (range: 1-5), the distance calculation will be almost entirely determined by price, ignoring bedrooms. In neural networks, features with larger magnitudes can cause unstable gradients and slow convergence. Normalization (L2) or standardization (z-score) puts all features on comparable scales, allowing the model to learn the true importance of each feature rather than being biased by their original scales. This leads to faster training and better generalization.',
    keyPoints: [
      'Unnormalized features: larger-scale features dominate distance/gradient calculations',
      'Normalization ensures all features contribute equally',
      'Leads to faster training and better generalization in ML models',
    ],
  },
  {
    id: 'vec-fund-d2',
    question:
      'How do vectors enable us to work with high-dimensional data that we cannot visualize? Discuss the relationship between geometric intuition from 2D/3D and mathematical operations in higher dimensions.',
    sampleAnswer:
      'Vectors allow us to extend geometric intuition to arbitrarily high dimensions through algebraic operations. While we can only visualize up to 3D, the mathematical operations (addition, scaling, dot products, distances) work identically in any dimension. For example, the distance formula in 2D (Pythagorean theorem) extends naturally to n dimensions. This is powerful because real-world ML problems often involve hundreds or thousands of dimensions (features). A document might be represented as a 10,000-dimensional vector (one per word), yet we can still compute distances, similarities, and perform optimization. The key insight is that geometric concepts like "angle between vectors," "distance," and "projection" have precise algebraic definitions that work in any dimension, even when we cannot draw them. This mathematical abstraction is what makes modern ML possible.',
    keyPoints: [
      'Algebraic operations extend geometric intuition to high dimensions',
      'Distance, angle, and projection have precise definitions in n-dimensions',
      'Enables ML to work with thousands of features (document vectors, embeddings)',
    ],
  },
  {
    id: 'vec-fund-d3',
    question:
      'In the word embedding example (king - man + woman ≈ queen), what does this vector arithmetic represent conceptually? How do vectors capture semantic relationships between words?',
    sampleAnswer:
      'This vector arithmetic captures semantic relationships and analogies in language. When we compute "king - man + woman," we are performing conceptual reasoning: take the concept of "king," remove the "male" aspect, and add the "female" aspect, yielding "queen." This works because word embeddings (like Word2Vec or GloVe) are trained so that words appearing in similar contexts have similar vectors. During training, the model learns that "king" and "man" often appear in male contexts, while "queen" and "woman" appear in female contexts. The vector differences encode relationships: "king - man" captures "royalty minus maleness," and adding "woman" gives "royalty plus femaleness" = "queen." This demonstrates that vectors can encode not just individual meanings but also relationships and transformations between concepts. It is a remarkable example of how continuous representations (vectors) can capture discrete semantic knowledge, enabling machines to reason about language mathematically.',
    keyPoints: [
      'Vector arithmetic captures semantic analogies: king - man + woman ≈ queen',
      'Word embeddings learn from context: similar contexts → similar vectors',
      'Vectors encode relationships and transformations, not just meanings',
    ],
  },
];
