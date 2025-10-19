/**
 * Quiz: K Means Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const k_means_clusteringQuiz: QuizQuestion[] = [
  {
    id: 'k-means-clustering-q1',
    question: `Explain the K-Means algorithm step-by-step. What are the key assumptions K-Means makes about cluster structure, and how do these assumptions limit its applicability?`,
    hint: 'Cover initialization, assignment, update steps, and assumptions about cluster shapes.',
    sampleAnswer: `K-Means algorithm: (1) Initialize K centroids randomly, (2) Assign each point to nearest centroid (Euclidean distance), (3) Update centroids to mean of assigned points, (4) Repeat steps 2-3 until convergence (centroids stop moving). Key assumptions: (1) SPHERICAL CLUSTERS: assumes clusters are roughly circular/spherical - fails on non-convex shapes like crescents, (2) SIMILAR SIZE: tends to create equal-sized clusters even if true clusters vary, (3) SIMILAR DENSITY: struggles with varying density clusters, (4) EUCLIDEAN DISTANCE: assumes isotropic variance - fails if features have different scales or correlations, (5) NUMBER K KNOWN: requires specifying K upfront. Limitations: Cannot find moon-shaped clusters, elongated clusters, or clusters with holes. Use DBSCAN for arbitrary shapes, hierarchical clustering if K unknown, GMM for probabilistic assignments.`,
    keyPoints: [
      'Algorithm: initialize centroids, assign points, update centroids, repeat',
      'Assumes spherical, similar-sized clusters',
      'Uses Euclidean distance (sensitive to scaling)',
      'Requires K specified upfront',
      'Fails on non-convex shapes and varying densities',
    ],
  },
  {
    id: 'k-means-clustering-q2',
    question: `What is the elbow method for choosing K in K-Means? Why might it sometimes fail, and what alternative methods can be used to determine the optimal number of clusters?`,
    hint: 'Explain WCSS, the elbow visualization, and discuss silhouette scores and other metrics.',
    sampleAnswer: `Elbow method plots Within-Cluster Sum of Squares (WCSS) vs K. WCSS always decreases as K increases, but at diminishing rates. The 'elbow' - where improvement slows dramatically - suggests optimal K. LIMITATIONS: (1) Elbow may be ambiguous or not exist, (2) Subjective interpretation, (3) Doesn't account for cluster validity. ALTERNATIVES: (1) SILHOUETTE SCORE: measures how similar points are to their own cluster vs other clusters (-1 to 1, higher better). Clear maximum indicates optimal K. (2) DAVIES-BOULDIN INDEX: ratio of within-cluster to between-cluster distances (lower better). (3) GAP STATISTIC: compares WCSS to expected WCSS under null distribution. (4) CROSS-VALIDATION: if downstream task exists, choose K based on task performance. (5) DOMAIN KNOWLEDGE: sometimes K is known from business context. Best practice: try multiple methods and validate with domain experts.`,
    keyPoints: [
      'Elbow method: plot WCSS vs K, look for bend',
      'Can be ambiguous or non-existent',
      'Silhouette score measures cluster cohesion and separation',
      'Multiple methods exist (Gap statistic, Davies-Bouldin)',
      'Validate with domain knowledge and downstream performance',
    ],
  },
  {
    id: 'k-means-clustering-q3',
    question: `Explain the K-Means++ initialization algorithm. Why is it superior to random initialization, and how does it work mathematically?`,
    hint: 'Cover the probability-based selection process and its advantages.',
    sampleAnswer: `K-Means++ is a smart initialization that improves convergence and final results. ALGORITHM: (1) Choose first centroid uniformly at random from data points, (2) For each remaining centroid: calculate distance from each point to nearest chosen centroid, choose next centroid with probability proportional to distance squared: P(x) = D(x)²/Σ D(x)², (3) Repeat until K centroids chosen. ADVANTAGES: (1) SPREADS CENTROIDS: probabilistic selection ensures centroids start far apart, (2) FASTER CONVERGENCE: fewer iterations needed, (3) BETTER RESULTS: reduces chance of poor local minima, (4) THEORETICAL GUARANTEE: solution is O(log K) competitive with optimal. INTUITION: Points far from existing centroids have higher probability of being chosen, leading to better initial coverage. This is now the default in sklearn. Random initialization can lead to poor results that K-Means++ avoids by starting with well-distributed centroids.`,
    keyPoints: [
      'Chooses initial centroids probabilistically, not randomly',
      'Probability proportional to distance squared from nearest centroid',
      'Ensures centroids start far apart',
      'Faster convergence and better results',
      'Now default in most implementations',
    ],
  },
];
