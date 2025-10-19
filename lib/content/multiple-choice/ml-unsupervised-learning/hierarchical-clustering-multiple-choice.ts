/**
 * Multiple Choice Questions: Hierarchical Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hierarchical_clusteringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'hierarchical-clustering-mc1',
    question: 'What is the time complexity of hierarchical clustering?',
    options: ['O(n)', 'O(n log n)', 'O(n² log n)', 'O(n³)'],
    correctAnswer: 2,
    explanation:
      'Hierarchical clustering has O(n² log n) time complexity with optimizations (naive implementation is O(n³)), and O(n²) space complexity for the distance matrix. This makes it impractical for very large datasets.',
    difficulty: 'medium',
  },
  {
    id: 'hierarchical-clustering-mc2',
    question: 'In hierarchical clustering, Ward linkage minimizes:',
    options: [
      'The minimum distance between clusters',
      'The maximum distance between clusters',
      'The within-cluster variance',
      'The number of merges needed',
    ],
    correctAnswer: 2,
    explanation:
      'Ward linkage merges clusters to minimize the increase in within-cluster variance, similar to the K-Means objective. This tends to create compact, spherical clusters of similar size.',
    difficulty: 'medium',
  },
  {
    id: 'hierarchical-clustering-mc3',
    question: 'What advantage does hierarchical clustering have over K-Means?',
    options: [
      "It's faster for large datasets",
      "It doesn't require specifying the number of clusters upfront",
      'It can only find spherical clusters',
      'It uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'Hierarchical clustering creates a dendrogram showing relationships at all levels, allowing you to choose the number of clusters after seeing the hierarchy. K-Means requires specifying K before running the algorithm.',
    difficulty: 'easy',
  },
  {
    id: 'hierarchical-clustering-mc4',
    question: 'In a dendrogram, what does the height of a merge represent?',
    options: [
      'The number of points in the cluster',
      'The distance or dissimilarity between the clusters being merged',
      'The order in which clusters were formed',
      'The variance within the cluster',
    ],
    correctAnswer: 1,
    explanation:
      'The height (y-axis) of a merge in a dendrogram represents the distance or dissimilarity between the clusters being merged. Larger heights indicate merging more dissimilar clusters, suggesting a good place to cut.',
    difficulty: 'easy',
  },
  {
    id: 'hierarchical-clustering-mc5',
    question:
      'You want to cluster 100,000 customer records. Which approach is most practical?',
    options: [
      'Use agglomerative hierarchical clustering directly on all 100,000 records',
      'Use K-Means or mini-batch K-Means for scalability',
      'Use complete linkage hierarchical clustering',
      'Hierarchical clustering is the best choice for large datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Hierarchical clustering has O(n² log n) time and O(n²) space complexity, making it impractical for 100,000 points (would need ~40GB just for the distance matrix). K-Means or mini-batch K-Means scales much better to large datasets. Alternatively, you could use hierarchical clustering on a sample, or use divisive methods, but standard agglomerative hierarchical clustering is not suitable.',
    difficulty: 'hard',
  },
];
