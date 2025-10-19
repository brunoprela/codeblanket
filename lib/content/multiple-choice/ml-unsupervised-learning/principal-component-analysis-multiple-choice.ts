/**
 * Multiple Choice Questions: Principal Component Analysis
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const principal_component_analysisMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'principal-component-analysis-mc1',
      question: 'What do the principal components in PCA represent?',
      options: [
        'The original features in descending order of importance',
        'New orthogonal axes that maximize variance',
        'The cluster centers of the data',
        'The outliers in the dataset',
      ],
      correctAnswer: 1,
      explanation:
        'Principal components are new orthogonal (perpendicular) axes that are linear combinations of original features. They are ordered by the amount of variance they explain, with PC1 explaining the most variance.',
    },
    {
      id: 'principal-component-analysis-mc2',
      question:
        'Why is feature scaling (standardization) crucial before applying PCA?',
      options: [
        "PCA doesn't work on unscaled data",
        'To make the algorithm run faster',
        'To prevent features with larger scales from dominating the principal components',
        'To ensure all eigenvalues are positive',
      ],
      correctAnswer: 2,
      explanation:
        'PCA is sensitive to feature scales because it uses distances/variance. Features with larger scales (e.g., salary in dollars vs age in years) will dominate the principal components if not scaled, leading to biased results.',
    },
    {
      id: 'principal-component-analysis-mc3',
      question: 'What information does the explained variance ratio provide?',
      options: [
        'The percentage of outliers in each component',
        'The proportion of total variance captured by each principal component',
        'The correlation between original features',
        'The number of samples in each cluster',
      ],
      correctAnswer: 1,
      explanation:
        'Explained variance ratio shows what percentage of the total variance in the data is captured by each principal component. This helps determine how many components to retain for dimensionality reduction.',
    },
    {
      id: 'principal-component-analysis-mc4',
      question: 'When should you NOT use PCA for dimensionality reduction?',
      options: [
        'When features are highly correlated',
        'When you need interpretable features',
        'When you have high-dimensional data',
        'When you want to visualize data',
      ],
      correctAnswer: 1,
      explanation:
        'Avoid PCA when interpretability is crucial, as principal components are linear combinations of original features and lose direct interpretability. Also avoid it when relationships are non-linear (use Kernel PCA or t-SNE instead).',
    },
  ];
