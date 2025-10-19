/**
 * Multiple choice questions for Multivariate Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const multivariateanalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'You find two features with correlation r = 0.92. Which type of model is MOST negatively affected by this multicollinearity?',
    options: [
      'Random Forest',
      'Linear Regression',
      'XGBoost',
      'K-Nearest Neighbors',
    ],
    correctAnswer: 1,
    explanation:
      'Linear regression is most affected by multicollinearity. It makes coefficients unstable and difficult to interpret. Tree-based models (Random Forest, XGBoost) handle multicollinearity well by naturally selecting one feature at splits.',
  },
  {
    id: 'mc2',
    question:
      'In PCA, what does it mean if the first principal component explains 60% of the variance?',
    options: [
      'The first component is 60% accurate',
      'The first component captures 60% of the variability in the original data',
      '60% of features are important',
      'The data has 60% noise',
    ],
    correctAnswer: 1,
    explanation:
      'Explained variance indicates how much of the total variability in the original features is captured by that principal component. 60% means the first PC represents more than half of all variation in the data.',
  },
  {
    id: 'mc3',
    question: 'Before applying PCA, you should:',
    options: [
      'Remove outliers only',
      'Standardize/scale all features to have mean=0 and std=1',
      'Convert all features to categorical',
      'Apply log transformation to all features',
    ],
    correctAnswer: 1,
    explanation:
      'PCA is sensitive to feature scales because it maximizes variance. Features with larger scales will dominate the principal components. Always standardize (or normalize) features before PCA so all features contribute fairly.',
  },
  {
    id: 'mc4',
    question:
      'A correlation heatmap shows a cluster of 5 features all highly correlated (r > 0.85) with each other. What is the BEST action?',
    options: [
      'Keep all 5 features - more data is always better',
      'Remove all 5 features',
      'Keep one feature from the cluster or combine them (e.g., with PCA or averaging)',
      'Apply log transformation to all 5',
    ],
    correctAnswer: 2,
    explanation:
      'Highly correlated features are redundant and can cause multicollinearity. Best practice is to keep the most interpretable/important feature from the cluster, or combine them into a single feature using PCA, averaging, or domain knowledge. Keeping all wastes computational resources and hurts linear models.',
  },
  {
    id: 'mc5',
    question:
      'What is the main advantage of a pair plot over a simple correlation matrix?',
    options: [
      'Pair plots are faster to compute',
      'Pair plots show actual data distributions and can reveal non-linear relationships that correlation might miss',
      'Pair plots only work with categorical data',
      'Pair plots are more accurate',
    ],
    correctAnswer: 1,
    explanation:
      "Correlation (especially Pearson) only measures linear relationships. Pair plots show the actual scatter plots, revealing non-linear patterns, outliers, clusters, and relationship shapes that a single correlation number misses. Always visualize, don't just rely on numbers!",
  },
];
