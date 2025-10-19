/**
 * Quiz: Other Dimensionality Reduction
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const other_dimensionality_reductionQuiz: QuizQuestion[] = [
  {
    id: 'other-dimensionality-reduction-q1',
    question: `Compare t-SNE and UMAP for dimensionality reduction. What are the key algorithmic differences, and in what scenarios would you choose one over the other?`,
    hint: 'Cover computational efficiency, global vs local structure, and the ability to transform new data.',
    sampleAnswer: `t-SNE: Converts high-D and low-D distances to probabilities, minimizes KL divergence. Preserves LOCAL structure (neighborhoods). UMAP: Based on Riemannian geometry and topological data analysis. Preserves BOTH local and global structure. KEY DIFFERENCES: (1) SPEED: UMAP much faster (can handle millions of points vs thousands for t-SNE), (2) GLOBAL STRUCTURE: UMAP preserves, t-SNE doesn't. UMAP better shows relationships between clusters. (3) NEW DATA: UMAP has .transform() method for new points. t-SNE must re-run entire algorithm. (4) STABILITY: UMAP more stable across runs. t-SNE very sensitive to random initialization. (5) PARAMETERS: t-SNE (perplexity, iterations). UMAP (n_neighbors, min_dist). WHEN TO USE t-SNE: Publication-quality visualizations, small datasets (<10K), only care about local structure. WHEN TO USE UMAP: Large datasets, need to transform new data, want global structure, general-purpose use. PRACTICAL: Start with UMAP for most use cases. Use t-SNE for beautiful visualizations of small datasets. Both better than PCA for non-linear structure.`,
    keyPoints: [
      't-SNE: preserves local structure, slow, no transform for new data',
      'UMAP: preserves both local and global structure, fast',
      'UMAP can transform new data, t-SNE cannot',
      'UMAP more scalable (millions vs thousands)',
      't-SNE best for visualization; UMAP for general use',
    ],
  },
  {
    id: 'other-dimensionality-reduction-q2',
    question: `What do t-SNE and UMAP visualizations tell us, and what don't they tell us? Explain common misinterpretations and best practices for interpreting these dimensionality reduction plots.`,
    hint: 'Cover cluster sizes, inter-cluster distances, and the limitations of 2D projections.',
    sampleAnswer: `WHAT THEY SHOW: (1) Which points are similar (close in plot = similar in high-D), (2) Rough cluster structure (groups that exist), (3) Relative neighborhood relationships. WHAT THEY DON'T SHOW (CRITICAL): (1) CLUSTER SIZE: Expansion/contraction is arbitrary. Large cluster in plot doesn't mean more points or higher density. (2) INTER-CLUSTER DISTANCE: Distance between clusters is meaningless. Two clusters close in plot may be far in high-D, or vice versa. (3) ABSOLUTE DISTANCES: Only local neighborhoods preserved. (4) AXES: No interpretable meaning (unlike PCA components). COMMON MISINTERPRETATIONS: 'Cluster A is bigger/denser than B' (No!), 'Clusters A and B are more similar than A and C' (No!), 'These two subclusters should merge' (Not enough info). BEST PRACTICES: (1) Run with multiple random seeds (especially t-SNE), (2) Try different perplexity/n_neighbors, (3) Use color to show known labels if available, (4) Validate clusters with domain knowledge, (5) Combine with other methods (hierarchical, silhouette scores), (6) Don't make quantitative claims from visualization alone. These are VISUALIZATION tools, not analysis tools. Use for exploration, not conclusions.`,
    keyPoints: [
      'Show: which points are similar, rough cluster structure',
      `Don't show: cluster sizes, inter-cluster distances, absolute distances`,
      'Common error: interpreting cluster size or distance between clusters',
      'Best practice: try multiple parameters, validate with other methods',
      'Visualization tools, not definitive analysis',
    ],
  },
  {
    id: 'other-dimensionality-reduction-q3',
    question: `Explain the concept of manifold learning. How do techniques like Isomap, LLE, and t-SNE/UMAP assume data lies on a low-dimensional manifold embedded in high-dimensional space?`,
    hint: 'Cover the manifold hypothesis and how different algorithms exploit it.',
    sampleAnswer: `MANIFOLD HYPOTHESIS: High-dimensional data often lies on or near a low-dimensional manifold (smooth surface) embedded in the high-D space. EXAMPLE: Images of a face rotated 360° appear high-D (pixel space) but actually lie on a 2D manifold (rotation parameters). MANIFOLD LEARNING: Algorithms that discover and represent this manifold structure. APPROACHES: (1) ISOMAP: Uses geodesic distances (distances along manifold surface, not Euclidean). Builds nearest-neighbor graph, computes shortest paths, applies MDS. Assumes single connected manifold. Good for: curved surfaces like swiss roll. (2) LLE (Locally Linear Embedding): Assumes manifold locally linear. Represents each point as weighted combination of neighbors, preserves these weights in low-D. Good for: smooth manifolds. (3) t-SNE/UMAP: Model probability of points being neighbors in high-D and low-D, minimize divergence. Don't explicitly model manifold but implicitly preserve manifold structure. WHY IT WORKS: Real data has structure - not random points in high-D space. Correlations and constraints create lower intrinsic dimensionality. Face images: determined by face parameters (pose, expression), not independent pixels. Manifold learning reveals true degrees of freedom.`,
    keyPoints: [
      'High-D data often lies on low-D manifold',
      'Isomap: geodesic distances along manifold surface',
      'LLE: locally linear structure preservation',
      't-SNE/UMAP: probabilistic neighborhood modeling',
      'Exploits data structure to find true dimensionality',
    ],
  },
];
