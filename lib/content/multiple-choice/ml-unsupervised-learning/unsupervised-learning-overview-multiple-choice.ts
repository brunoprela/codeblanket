/**
 * Multiple Choice Questions: Unsupervised Learning Overview
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const unsupervised_learning_overviewMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'unsupervised-learning-overview-mc1',
      question:
        'What is the primary difference between supervised and unsupervised learning?',
      options: [
        'Supervised learning is faster than unsupervised learning',
        'Supervised learning requires labeled data, while unsupervised learning does not',
        'Unsupervised learning is more accurate than supervised learning',
        'Supervised learning works only with numerical data',
      ],
      correctAnswer: 1,
      explanation:
        'The fundamental difference is that supervised learning requires labeled training data (input-output pairs), while unsupervised learning discovers patterns in unlabeled data without predefined target variables.',
    },
    {
      id: 'unsupervised-learning-overview-mc2',
      question:
        'Which of the following is NOT a main category of unsupervised learning?',
      options: [
        'Clustering',
        'Classification',
        'Dimensionality Reduction',
        'Anomaly Detection',
      ],
      correctAnswer: 1,
      explanation:
        'Classification is a supervised learning task that requires labeled data. The three main categories of unsupervised learning are clustering, dimensionality reduction, and anomaly detection.',
    },
    {
      id: 'unsupervised-learning-overview-mc3',
      question: 'The curse of dimensionality refers to:',
      options: [
        'The difficulty of storing high-dimensional data',
        'The phenomenon where data becomes sparse and distances less meaningful in high dimensions',
        'The inability to visualize data beyond 3 dimensions',
        'The requirement for more computational power',
      ],
      correctAnswer: 1,
      explanation:
        'The curse of dimensionality describes how, as dimensionality increases, data becomes increasingly sparse, distances between points become less meaningful, and algorithms become less effective. This fundamentally affects how unsupervised learning algorithms work.',
    },
    {
      id: 'unsupervised-learning-overview-mc4',
      question:
        'A company has millions of customer transaction records without any labels. They want to identify groups of customers with similar purchasing behaviors. Which unsupervised learning task is most appropriate?',
      options: [
        'Regression',
        'Clustering',
        'Classification',
        'Time series forecasting',
      ],
      correctAnswer: 1,
      explanation:
        'Clustering is the appropriate unsupervised learning task for grouping customers with similar purchasing behaviors. It will discover natural segments in the data without requiring predefined labels. Regression and classification are supervised tasks, and time series forecasting focuses on temporal predictions rather than grouping.',
    },
    {
      id: 'unsupervised-learning-overview-mc5',
      question:
        'You have a dataset with 1000 features and want to visualize it in 2D to identify patterns. Which type of unsupervised learning technique would you use?',
      options: [
        'Clustering algorithms like K-Means',
        'Dimensionality reduction techniques like PCA or t-SNE',
        'Anomaly detection methods like Isolation Forest',
        'Association rule mining like Apriori',
      ],
      correctAnswer: 1,
      explanation:
        'Dimensionality reduction techniques like PCA (Principal Component Analysis) or t-SNE are specifically designed to reduce high-dimensional data to 2D or 3D for visualization while preserving important structure. While clustering might be applied after dimensionality reduction, the core task of reducing 1000 features to 2D requires dimensionality reduction.',
    },
  ];
