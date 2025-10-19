/**
 * Multiple Choice Questions: Other Dimensionality Reduction
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const other_dimensionality_reductionMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'other-dimensionality-reduction-mc1',
      question: 'What is the main difference between PCA and t-SNE?',
      options: [
        'PCA is supervised, t-SNE is unsupervised',
        'PCA is linear, t-SNE is non-linear',
        'PCA is slower than t-SNE',
        'PCA requires more parameters than t-SNE',
      ],
      correctAnswer: 1,
      explanation:
        'PCA is a linear dimensionality reduction technique that finds linear combinations of features, while t-SNE is non-linear and can capture complex manifold structures, making it better for visualization of complex data.',
      difficulty: 'easy',
    },
    {
      id: 'other-dimensionality-reduction-mc2',
      question: 'Which statement about t-SNE visualizations is TRUE?',
      options: [
        'The size of clusters indicates the number of points in each cluster',
        'The distance between clusters is meaningful and can be interpreted',
        'Points close together in the plot are similar in the high-dimensional space',
        'The axes have interpretable meanings like in PCA',
      ],
      correctAnswer: 2,
      explanation:
        'In t-SNE, points close together ARE similar in high-D space (local structure preserved). However, cluster sizes, inter-cluster distances, and axes are NOT meaningful or interpretable.',
      difficulty: 'medium',
    },
    {
      id: 'other-dimensionality-reduction-mc3',
      question: 'What is a key advantage of UMAP over t-SNE?',
      options: [
        'UMAP is simpler to understand',
        'UMAP preserves global structure better and can transform new data',
        'UMAP requires no parameters',
        'UMAP works only on small datasets',
      ],
      correctAnswer: 1,
      explanation:
        'UMAP preserves both local and global structure (unlike t-SNE which focuses on local), is faster, more scalable, and crucially has a .transform() method to project new data points without retraining.',
      difficulty: 'medium',
    },
    {
      id: 'other-dimensionality-reduction-mc4',
      question: 'The manifold hypothesis in machine learning states that:',
      options: [
        'All data is linearly separable',
        'High-dimensional data often lies on or near a low-dimensional manifold',
        'Dimensionality reduction always improves model performance',
        'Data must be normalized before analysis',
      ],
      correctAnswer: 1,
      explanation:
        'The manifold hypothesis suggests that real-world high-dimensional data often lies on or near a much lower-dimensional manifold (smooth surface) embedded in the high-dimensional space, which justifies dimensionality reduction.',
      difficulty: 'medium',
    },
    {
      id: 'other-dimensionality-reduction-mc5',
      question:
        'You need to visualize high-dimensional customer data in 2D to present to stakeholders AND later project new customers onto the same map. Which technique is most appropriate?',
      options: [
        't-SNE because it produces the best visualizations',
        'PCA because it preserves maximum variance',
        'UMAP because it provides good visualizations and has a transform method for new data',
        'Hierarchical clustering for better interpretability',
      ],
      correctAnswer: 2,
      explanation:
        "UMAP is ideal here because it produces high-quality visualizations (comparable to t-SNE) AND has a .transform() method to project new data onto the same embedding. t-SNE lacks this capability - you'd need to retrain on all data including new points. PCA is less suitable for visualization of complex patterns. Clustering is not a dimensionality reduction technique.",
      difficulty: 'hard',
    },
  ];
