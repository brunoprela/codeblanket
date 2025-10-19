/**
 * Quiz: Unsupervised Learning Overview
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const unsupervised_learning_overviewQuiz: QuizQuestion[] = [
  {
    id: 'unsupervised-learning-overview-q1',
    question: `What are the fundamental differences between supervised and unsupervised learning, and in what scenarios would you choose unsupervised learning over supervised learning?`,
    hint: 'Consider the availability of labels, the goal of analysis, and typical applications.',
    sampleAnswer: `Supervised learning requires labeled training data where both inputs and desired outputs are known, while unsupervised learning works with unlabeled data to discover hidden patterns. Unsupervised learning is chosen when: (1) Labels are unavailable, expensive, or impractical to obtain, (2) The goal is exploratory data analysis or pattern discovery rather than prediction, (3) You want to reduce dimensionality before supervised learning, (4) You need to detect anomalies or outliers, (5) You want to segment customers or group similar items without predefined categories. Supervised learning is preferred when you have clear target variables and sufficient labeled data for training.`,
    keyPoints: [
      'Supervised learning requires labeled data; unsupervised does not',
      'Unsupervised is used for pattern discovery and exploration',
      'Common when labels are expensive or unavailable',
      'Often used for preprocessing (dimensionality reduction)',
      'Suitable for clustering, anomaly detection, and segmentation',
    ],
  },
  {
    id: 'unsupervised-learning-overview-q2',
    question: `Explain the curse of dimensionality and its implications for unsupervised learning algorithms. How do dimensionality reduction techniques help mitigate this problem?`,
    hint: 'Consider distance metrics, data sparsity, and computational complexity.',
    sampleAnswer: `The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces. As dimensions increase: (1) Distances become less meaningful - all points appear equally far apart, (2) Data becomes increasingly sparse - exponentially more data needed to maintain density, (3) Computational complexity explodes - algorithms scale poorly, (4) Visualization becomes impossible beyond 3D. For unsupervised learning, this affects clustering (distances lose meaning), anomaly detection (everything looks like an outlier), and pattern discovery. Dimensionality reduction techniques like PCA, t-SNE, and UMAP help by: (1) Removing redundant/correlated features, (2) Projecting to lower dimensions while preserving structure, (3) Reducing computational cost, (4) Enabling visualization, (5) Removing noise. This makes algorithms more effective and interpretable.`,
    keyPoints: [
      'High dimensions make distances less meaningful',
      'Data becomes increasingly sparse',
      'Computational complexity increases exponentially',
      'Affects clustering, anomaly detection, visualization',
      'Dimensionality reduction preserves structure while reducing dimensions',
    ],
  },
  {
    id: 'unsupervised-learning-overview-q3',
    question: `Compare and contrast the three main categories of unsupervised learning: clustering, dimensionality reduction, and anomaly detection. Provide real-world examples for each.`,
    hint: 'Focus on objectives, outputs, and typical use cases.',
    sampleAnswer: `The three main categories serve different purposes: CLUSTERING groups similar items together, producing discrete group assignments. Output is cluster labels. Used for: customer segmentation (grouping customers by behavior), document organization (grouping similar articles), image segmentation (grouping pixels). DIMENSIONALITY REDUCTION transforms high-D data to lower-D while preserving structure. Output is reduced feature set or projection. Used for: visualization (plotting high-D data in 2D/3D), preprocessing (reducing features before supervised learning), compression (storing images efficiently). ANOMALY DETECTION identifies unusual patterns. Output is anomaly scores or binary flags. Used for: fraud detection (identifying suspicious transactions), intrusion detection (finding network attacks), quality control (detecting defective products). They can be combined: reduce dimensions with PCA, then cluster with K-Means, then detect anomalies in clusters.`,
    keyPoints: [
      'Clustering: groups similar items, outputs cluster labels',
      'Dimensionality reduction: reduces features, outputs projections',
      'Anomaly detection: finds outliers, outputs anomaly scores',
      'Each serves distinct purposes with different applications',
      'Can be combined for comprehensive analysis',
    ],
  },
];
