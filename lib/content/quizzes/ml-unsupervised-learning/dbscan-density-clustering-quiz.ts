/**
 * Quiz: Dbscan Density Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const dbscan_density_clusteringQuiz: QuizQuestion[] = [
  {
    id: 'dbscan-density-clustering-q1',
    question: `Explain how DBSCAN identifies clusters without requiring the number of clusters K as input. What are core points, border points, and noise points, and how are they determined?`,
    hint: 'Cover the epsilon and MinPts parameters and how they define density.',
    sampleAnswer: `DBSCAN discovers clusters as high-density regions separated by low-density regions, without needing K upfront. PARAMETERS: Epsilon (ε) = neighborhood radius; MinPts = minimum points to form dense region. POINT TYPES: (1) CORE POINT: has ≥ MinPts neighbors within ε distance (including itself). Forms cluster backbone. (2) BORDER POINT: has < MinPts neighbors but is within ε of a core point. Belongs to cluster but doesn't expand it. (3) NOISE POINT: neither core nor border. In low-density regions, marked as outliers. ALGORITHM: (1) For each unvisited point, count neighbors within ε, (2) If ≥ MinPts, start new cluster and recursively add all density-connected points, (3) If < MinPts but near core point, mark as border of that cluster, (4) Otherwise mark as noise. NUMBER OF CLUSTERS: determined by data density structure, not predefined. A point is density-reachable from another if connected through core points. This allows arbitrary cluster shapes unlike K-Means.`,
    keyPoints: [
      'Discovers clusters based on density, not predefined K',
      'Core points: ≥ MinPts neighbors within ε',
      'Border points: in ε-neighborhood of core point',
      'Noise points: neither core nor border, marked as outliers',
      'Number of clusters determined by data structure',
    ],
  },
  {
    id: 'dbscan-density-clustering-q2',
    question: `How do you choose appropriate values for epsilon (ε) and MinPts in DBSCAN? Explain the k-distance graph method and provide practical guidelines.`,
    hint: 'Cover the elbow in k-distance plot and rules of thumb for MinPts.',
    sampleAnswer: `CHOOSING MinPts: RULE OF THUMB: MinPts ≥ dimensions + 1. For 2D: MinPts ≥ 3. Common values: 4, 5, 10. Increase for noisy data (more strict), decrease for sparse data. Start with MinPts = 5 as default. CHOOSING EPSILON: Use K-DISTANCE GRAPH: (1) For each point, compute distance to k-th nearest neighbor (k = MinPts), (2) Sort distances ascending, (3) Plot sorted distances, (4) Look for 'elbow' - where curve sharply increases, (5) Points before elbow are in dense regions, after are outliers. Elbow height suggests good ε. INTUITION: Below ε, points in clusters; above ε, in sparse regions. PRACTICAL TIPS: (1) Scale features first (DBSCAN uses distance), (2) Try multiple ε values around elbow, (3) Use silhouette score to compare, (4) Domain knowledge: what distance makes sense?, (5) Start with ε that gives reasonable number of clusters (2-10). PROBLEM: Single ε struggles with varying density - consider HDBSCAN for adaptive ε.`,
    keyPoints: [
      'MinPts ≥ dimensions + 1, typically 4-10',
      'Use k-distance graph to find epsilon',
      'Elbow in sorted k-distances suggests good ε',
      'Must scale features (distance-based)',
      'Single ε struggles with varying density',
    ],
  },
  {
    id: 'dbscan-density-clustering-q3',
    question: `Compare DBSCAN with K-Means and Hierarchical clustering. In what scenarios does DBSCAN excel, and when should you use alternatives instead?`,
    hint: 'Consider cluster shapes, scalability, parameter requirements, and handling of outliers.',
    sampleAnswer: `DBSCAN ADVANTAGES: (1) ARBITRARY SHAPES: finds non-spherical clusters (moons, spirals) that K-Means misses, (2) OUTLIER DETECTION: explicitly identifies noise points, (3) NO K REQUIRED: number of clusters determined automatically, (4) ROBUST: insensitive to initialization (deterministic). LIMITATIONS: (1) PARAMETER SENSITIVITY: ε and MinPts difficult to tune, (2) VARYING DENSITY: single ε cannot handle clusters of different densities, (3) HIGH DIMENSIONS: curse of dimensionality makes distances meaningless, (4) BORDER POINTS: assignment can be ambiguous. WHEN TO USE DBSCAN: Geospatial data, arbitrary-shaped clusters, need outlier detection, clusters have similar density. USE K-MEANS: Spherical clusters, fast result needed, very large datasets, K known. USE HIERARCHICAL: Small dataset, need dendrogram, K unknown, varying density OK. USE HDBSCAN: DBSCAN benefits + varying density. PERFORMANCE: DBSCAN O(n log n) with spatial index vs K-Means O(nKt) - DBSCAN can be faster if K large.`,
    keyPoints: [
      'DBSCAN: arbitrary shapes, outlier detection, no K needed',
      'K-Means: faster, spherical clusters, requires K',
      'Hierarchical: dendrogram, no K, but slow',
      'DBSCAN excels with non-spherical clusters and noise',
      'Use HDBSCAN for varying density',
    ],
  },
];
