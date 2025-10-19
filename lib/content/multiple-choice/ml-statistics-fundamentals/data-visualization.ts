/**
 * Multiple choice questions for Data Visualization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const datavisualizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which plot is most appropriate for visualizing the relationship between a continuous feature and a binary classification target?',
    options: [
      'Scatter plot',
      'Box plot (one for each class)',
      'Pie chart',
      'Line plot',
    ],
    correctAnswer: 1,
    explanation:
      'Box plots (one for each class) clearly show how the continuous feature distribution differs between classes. This reveals whether the feature can separate the classes. Scatter plot would need both axes continuous, pie charts are for categorical proportions, and line plots are for sequences.',
  },
  {
    id: 'mc2',
    question:
      'In a Q-Q (quantile-quantile) plot, if the points fall along the diagonal line, what does this indicate?',
    options: [
      'The data has high variance',
      'The data follows the reference distribution (e.g., normal)',
      'The data has outliers',
      'The data is discrete',
    ],
    correctAnswer: 1,
    explanation:
      'When points in a Q-Q plot fall along the diagonal line, the data quantiles match the theoretical distribution quantiles, indicating the data follows that distribution. Deviations from the line indicate departures from the assumed distribution.',
  },
  {
    id: 'mc3',
    question:
      'You have 8 features and want to visualize all pairwise correlations. What is the most efficient single visualization?',
    options: [
      '8 separate scatter plots',
      'A correlation heatmap',
      'A box plot for each feature',
      'A single scatter plot with all features',
    ],
    correctAnswer: 1,
    explanation:
      'A correlation heatmap shows all pairwise correlations in a single, color-coded matrix. It efficiently displays 8×8=64 relationships (or 28 unique pairs) at once. Scatter plots would require 28 separate plots for unique pairs, box plots show distribution not correlation, and you cannot plot 8 features on 2D scatter axes.',
  },
  {
    id: 'mc4',
    question:
      'In a learning curve, training score is high and validation score is low with a large gap. Adding more training data does NOT close the gap. What is the problem?',
    options: [
      'High bias (underfitting)',
      'High variance (overfitting)',
      'Perfect model',
      'Insufficient features',
    ],
    correctAnswer: 1,
    explanation:
      "A persistent large gap between high training score and low validation score indicates overfitting (high variance). The model memorizes training data but doesn't generalize. If more data doesn't help, the model has too much capacity and needs regularization or reduced complexity.",
  },
  {
    id: 'mc5',
    question:
      'Which visualization best shows the distribution of a categorical variable?',
    options: ['Histogram', 'Scatter plot', 'Bar chart', 'Line plot'],
    correctAnswer: 2,
    explanation:
      'Bar charts are ideal for categorical data, showing the count or proportion of each category. Histograms are for continuous variables, scatter plots for relationships between two continuous variables, and line plots for ordered sequences (like time series).',
  },
];
