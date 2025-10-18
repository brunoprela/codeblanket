/**
 * Multiple choice questions for Principal Component Analysis (PCA) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const principalcomponentanalysisMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pca-q1',
      question:
        'What do the principal components in PCA represent geometrically?',
      options: [
        'The mean of the data',
        'Orthogonal directions of maximum variance',
        'The features with highest correlation',
        'Random projections of the data',
      ],
      correctAnswer: 1,
      explanation:
        'Principal components are orthogonal (uncorrelated) directions in feature space ordered by the amount of variance they capture. PC1 points in the direction of maximum variance, PC2 in the direction of maximum remaining variance orthogonal to PC1, etc.',
    },
    {
      id: 'pca-q2',
      question:
        'Why is standardization (scaling features to mean=0, std=1) important before applying PCA?',
      options: [
        'It makes computation faster',
        'PCA is sensitive to feature scales and will be dominated by high-variance features',
        'It is required for SVD to work',
        'It improves interpretability',
      ],
      correctAnswer: 1,
      explanation:
        'PCA finds directions of maximum variance. If features have different scales (e.g., one in meters, another in kilometers), the high-scale feature will dominate the first principal component regardless of its importance. Standardization ensures all features are treated equally.',
    },
    {
      id: 'pca-q3',
      question:
        'If the first 3 principal components explain 95% of the variance, what happens to the remaining 5% when you project data onto these 3 components?',
      options: [
        'It is stored separately',
        'It is lost (cannot be recovered)',
        'It is distributed among the 3 components',
        'It becomes noise',
      ],
      correctAnswer: 1,
      explanation:
        'The remaining 5% of variance is discarded when projecting onto only 3 components. This is lossy compression - you can approximate the original data from the 3 components, but cannot perfectly reconstruct it. The reconstruction error equals the sum of discarded eigenvalues.',
    },
    {
      id: 'pca-q4',
      question:
        'Why might PCA via SVD be preferred over eigendecomposition of the covariance matrix?',
      options: [
        'SVD is always faster',
        'SVD works for non-square matrices',
        "SVD is more numerically stable and doesn't require forming the covariance matrix",
        'SVD gives different results',
      ],
      correctAnswer: 2,
      explanation:
        'SVD is more numerically stable because it avoids forming XᵀX, which squares the condition number and can lead to numerical errors. For tall matrices (many samples, fewer features), SVD is also more efficient. Both methods give the same principal components.',
    },
    {
      id: 'pca-q5',
      question:
        'PCA is limited to linear dimensionality reduction. Which statement best describes when to use alternative methods?',
      options: [
        'Always use PCA first regardless of data structure',
        'Use Kernel PCA or manifold learning (t-SNE, UMAP) when data lies on a nonlinear manifold',
        'Never use PCA for high-dimensional data',
        'PCA only works for 2D data',
      ],
      correctAnswer: 1,
      explanation:
        'PCA finds linear combinations of features. For data with nonlinear structure (e.g., Swiss roll, circles), PCA is inefficient or ineffective. Kernel PCA applies PCA in a high-dimensional feature space (nonlinear), while t-SNE and UMAP preserve local neighborhood structure for visualization.',
    },
  ];
