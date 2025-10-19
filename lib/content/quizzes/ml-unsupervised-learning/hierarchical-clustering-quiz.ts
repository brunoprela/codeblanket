/**
 * Quiz: Hierarchical Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const hierarchical_clusteringQuiz: QuizQuestion[] = [
  {
    id: 'hierarchical-clustering-q1',
    question: `Compare and contrast the four main linkage methods (single, complete, average, Ward) in hierarchical clustering. When would you choose each method, and what are their respective advantages and disadvantages?`,
    hint: 'Consider cluster shape preferences, sensitivity to outliers, and typical use cases.',
    sampleAnswer: `SINGLE LINKAGE (minimum distance): merges clusters with closest pair of points. Pros: finds non-elliptical shapes, can handle chains. Cons: sensitive to noise/outliers, prone to 'chaining' effect (long, snake-like clusters). Use for: non-convex shapes when data is clean. COMPLETE LINKAGE (maximum distance): merges clusters with farthest pair closest. Pros: creates compact, spherical clusters; less sensitive to outliers. Cons: can break large clusters; biased toward equal size. Use for: spherical clusters, noisy data. AVERAGE LINKAGE: merges based on average pairwise distance. Pros: balanced approach; more robust than single; less biased than complete. Cons: computationally expensive. Use for: general-purpose clustering when you want balance. WARD: minimizes within-cluster variance. Pros: similar objective to K-Means; creates balanced clusters; most commonly used. Cons: only works with Euclidean distance; biased toward equal-sized clusters. Use for: general-purpose clustering, default choice. Ward is most popular; single for special shapes; complete for noise robustness.`,
    keyPoints: [
      'Single: minimum distance, finds chains, sensitive to noise',
      'Complete: maximum distance, compact clusters, breaks large clusters',
      'Average: balanced compromise, robust, more expensive',
      'Ward: minimizes variance, most popular, similar to K-Means',
      'Choice depends on data characteristics and cluster shape expectations',
    ],
  },
  {
    id: 'hierarchical-clustering-q2',
    question: `How do you interpret a dendrogram and use it to choose the number of clusters? What information does the height of merges provide?`,
    hint: `Explain how to 'cut' the dendrogram and what vertical distances mean.`,
    sampleAnswer: `A dendrogram is a tree showing hierarchical relationships. READING: X-axis shows data points/clusters; Y-axis shows distance/dissimilarity at which clusters merge; horizontal lines represent clusters; vertical lines show merges; longer vertical lines = more dissimilar clusters being merged. CHOOSING K: Look for large vertical gaps - these indicate natural separations. Cut horizontally through the dendrogram: number of vertical lines you cross = number of clusters. HEIGHT INTERPRETATION: Height represents distance between clusters being merged. Large jump in height suggests merging very dissimilar clusters (don't merge). PRACTICAL APPROACH: (1) Look for longest vertical lines without horizontal crossings (biggest gaps), (2) Cut below these gaps, (3) Count resulting clusters. Can also use inconsistency method: measures how inconsistent a merge is compared to merges at adjacent levels. High inconsistency = good place to cut. The dendrogram visualizes clustering at ALL scales simultaneously, unlike K-Means which commits to one K.`,
    keyPoints: [
      'Dendrogram shows hierarchical clustering at all scales',
      'Y-axis height represents distance between merging clusters',
      'Large vertical gaps suggest natural cluster boundaries',
      'Cut horizontally to obtain K clusters',
      'Provides visual interpretation K-Means cannot offer',
    ],
  },
  {
    id: 'hierarchical-clustering-q3',
    question: `Explain the time and space complexity of hierarchical clustering. Why doesn't it scale well to large datasets, and what strategies can be used to apply it to larger data?`,
    hint: 'Cover distance matrix computation, algorithm complexity, and sampling strategies.',
    sampleAnswer: `COMPLEXITY: Time: O(n³) naive, O(n² log n) with optimizations. Space: O(n²) to store distance matrix. SCALABILITY ISSUES: (1) Must compute and store full distance matrix (n² space), (2) Must consider all pairs at each merge (expensive), (3) Cannot parallelize easily, (4) Impractical for n > 10,000 samples. STRATEGIES FOR LARGE DATA: (1) SAMPLING: cluster representative sample, then assign remaining points to nearest cluster (loses some structure), (2) PCA PREPROCESSING: reduce dimensions first (faster distances), (3) MINI-BATCH: divide data into batches, cluster each, then cluster the cluster centers (hierarchical of hierarchical), (4) USE ALTERNATIVE: switch to K-Means (O(nKt)) or DBSCAN for large datasets, (5) APPROXIMATE METHODS: use approximate nearest neighbors. WHEN TO USE HIERARCHICAL: Best for n < 10,000, when dendrogram is valuable, when K is unknown, for exploratory analysis. For large datasets, use K-Means or DBSCAN instead, or sample then apply hierarchical.`,
    keyPoints: [
      'O(n²) space for distance matrix',
      'O(n² log n) time complexity',
      'Not scalable beyond ~10,000 samples',
      'Can use sampling, preprocessing, or alternative algorithms',
      'Best for small-medium datasets when hierarchy is valuable',
    ],
  },
];
