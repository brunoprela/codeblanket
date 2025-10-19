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
      difficulty: 'easy',
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
      difficulty: 'medium',
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
      difficulty: 'easy',
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
      difficulty: 'medium',
    },
    {
      id: 'principal-component-analysis-mc5',
      question:
        'You apply PCA to a dataset with 100 features. The first 3 principal components explain 95% of the variance. What should you conclude?',
      options: [
        'The dataset has too many features and PCA failed',
        'The original 100 features are highly redundant; you can reduce to 3 components with minimal information loss',
        'You must keep all 100 features since some variance is lost',
        'PCA is not suitable for this dataset',
      ],
      correctAnswer: 1,
      explanation:
        'When the first few principal components capture most of the variance (e.g., 95%), it indicates that the original features are highly correlated/redundant. You can safely reduce dimensionality from 100 to 3 while retaining 95% of the information. This is one of the main benefits of PCA - discovering that high-dimensional data often lies on a lower-dimensional manifold.',
      difficulty: 'medium',
    },
  ];
