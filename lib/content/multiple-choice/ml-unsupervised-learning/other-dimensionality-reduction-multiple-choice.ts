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
    },
  ];
