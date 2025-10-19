/**
 * Multiple Choice Questions: K Means Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const k_means_clusteringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'k-means-clustering-mc1',
    question: 'In K-Means clustering, what does the algorithm minimize?',
    options: [
      'The maximum distance between any two points',
      'The within-cluster sum of squares (WCSS)',
      'The number of iterations needed for convergence',
      'The distance between cluster centroids',
    ],
    correctAnswer: 1,
    explanation:
      'K-Means aims to minimize the Within-Cluster Sum of Squares (WCSS), also called inertia - the sum of squared distances from each point to its assigned cluster centroid. This creates compact, spherical clusters.',
    difficulty: 'easy',
  },
  {
    id: 'k-means-clustering-mc2',
    question:
      'What is the main advantage of K-Means++ initialization over random initialization?',
    options: [
      'It requires fewer clusters',
      'It runs faster',
      'It chooses initial centroids that are spread apart, leading to better and faster convergence',
      'It can find non-spherical clusters',
    ],
    correctAnswer: 2,
    explanation:
      'K-Means++ uses a probabilistic method to select initial centroids that are far apart from each other, which leads to better final results and faster convergence compared to random initialization.',
    difficulty: 'medium',
  },
  {
    id: 'k-means-clustering-mc3',
    question: 'Which of the following is a limitation of K-Means?',
    options: [
      'It works only with categorical data',
      'It assumes clusters are spherical and of similar size',
      'It cannot handle more than 10 features',
      'It requires knowing the data labels',
    ],
    correctAnswer: 1,
    explanation:
      'K-Means assumes clusters are roughly spherical (convex), of similar size, and have similar density. It struggles with elongated, non-convex shapes (like crescent moons) and clusters of very different sizes.',
    difficulty: 'medium',
  },
  {
    id: 'k-means-clustering-mc4',
    question: 'What does the elbow method help determine in K-Means?',
    options: [
      'The optimal learning rate',
      'The optimal number of clusters (K)',
      'The optimal number of iterations',
      'The optimal distance metric',
    ],
    correctAnswer: 1,
    explanation:
      "The elbow method plots WCSS vs K and looks for the 'elbow' point where adding more clusters yields diminishing returns. This suggests the optimal number of clusters.",
    difficulty: 'easy',
  },
  {
    id: 'k-means-clustering-mc5',
    question:
      'You run K-Means with K=3 on a dataset and get WCSS=500. You then run it again with K=6 and get WCSS=200. What can you conclude?',
    options: [
      'K=6 is always better because it has lower WCSS',
      'K=3 is better because it uses fewer clusters',
      'Lower WCSS with more clusters is expected; you need additional metrics like silhouette score to decide',
      'The algorithm failed because WCSS should increase with more clusters',
    ],
    correctAnswer: 2,
    explanation:
      'WCSS always decreases as K increases (it reaches 0 when K=n, with each point as its own cluster). Lower WCSS does not automatically mean better clustering. You need to consider the elbow method, silhouette score, or domain knowledge to choose K. The goal is to balance cluster quality with simplicity.',
    difficulty: 'hard',
  },
];
