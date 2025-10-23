/**
 * Multiple Choice Questions: Dbscan Density Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const dbscan_density_clusteringMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dbscan-density-clustering-mc1',
      question: 'In DBSCAN, a core point is defined as:',
      options: [
        'A point at the center of a cluster',
        'A point with at least MinPts neighbors within epsilon distance',
        'A point that is an outlier',
        'The first point visited by the algorithm',
      ],
      correctAnswer: 1,
      explanation:
        'A core point in DBSCAN has at least MinPts neighbors (including itself) within epsilon (ε) distance. Core points form the backbone of clusters and can extend the cluster to other density-connected points.',
    },
    {
      id: 'dbscan-density-clustering-mc2',
      question: 'What is a key advantage of DBSCAN over K-Means?',
      options: [
        'DBSCAN is always faster',
        'DBSCAN can find arbitrarily shaped clusters and identify outliers',
        'DBSCAN requires less memory',
        'DBSCAN works better with very high-dimensional data',
      ],
      correctAnswer: 1,
      explanation:
        'DBSCAN can discover clusters of arbitrary shapes (not just spherical) and explicitly identifies outliers as noise points, unlike K-Means which forces every point into a cluster and assumes spherical shapes.',
    },
    {
      id: 'dbscan-density-clustering-mc3',
      question:
        'How do you choose an appropriate epsilon (ε) value for DBSCAN?',
      options: [
        'Always use ε = 1.0',
        'Use the k-distance graph and look for the elbow point',
        'Set ε equal to the number of clusters desired',
        'Use the average distance between all points',
      ],
      correctAnswer: 1,
      explanation:
        "The k-distance graph method plots sorted distances to the k-th nearest neighbor (k = MinPts). The 'elbow' point in this curve suggests a good epsilon value, separating dense regions from sparse regions.",
    },
    {
      id: 'dbscan-density-clustering-mc4',
      question: 'What is a major limitation of DBSCAN?',
      options: [
        'It cannot detect outliers',
        'It requires knowing K in advance',
        'It struggles with clusters of varying densities',
        'It only works with categorical data',
      ],
      correctAnswer: 2,
      explanation:
        'DBSCAN uses a single epsilon value, which makes it difficult to handle clusters with significantly different densities. HDBSCAN (Hierarchical DBSCAN) addresses this limitation by using varying epsilon values.',
    },
    {
      id: 'dbscan-density-clustering-mc5',
      question:
        'You have a spatial dataset with distinct clusters of different densities: a dense urban core and sparse suburban areas. What would be the best approach?',
      options: [
        'Use standard DBSCAN with epsilon tuned to the dense core',
        'Use K-Means since it handles varying densities better',
        'Use HDBSCAN or OPTICS which handle varying densities',
        'Use standard DBSCAN with epsilon tuned to the sparse regions',
      ],
      correctAnswer: 2,
      explanation:
        'HDBSCAN (Hierarchical DBSCAN) or OPTICS are designed to handle clusters of varying densities by using hierarchical approaches or ordered reachability plots. Standard DBSCAN with a single epsilon will either merge sparse clusters or fragment dense ones. K-Means also struggles with varying densities as it assumes spherical, equally-sized clusters.',
    },
  ];
