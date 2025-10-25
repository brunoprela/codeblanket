/**
 * Quiz: Principal Component Analysis
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const principal_component_analysisQuiz: QuizQuestion[] = [
  {
    id: 'principal-component-analysis-q1',
    question: `Explain the mathematical foundation of PCA. How does PCA use eigenvalue decomposition of the covariance matrix to find principal components, and what do eigenvalues and eigenvectors represent?`,
    hint: 'Cover covariance matrix, eigendecomposition, and the meaning of eigenvalues/eigenvectors.',
    sampleAnswer: `PCA finds orthogonal axes that maximize variance. MATHEMATICAL STEPS: (1) CENTER DATA: X_centered = X - mean, (2) COVARIANCE MATRIX: Σ = (1/n-1) X_centered^T @ X_centered. Measures how features vary together. (3) EIGENDECOMPOSITION: Σv = λv. Finds eigenvectors (v) and eigenvalues (λ). (4) SORT: order eigenvectors by eigenvalues (descending). (5) PROJECT: X_pca = X_centered @ V_k (top k eigenvectors). INTERPRETATION: EIGENVECTORS (principal components) = new axes, directions of maximum variance. Orthogonal to each other. Linear combinations of original features. EIGENVALUES = variance along each eigenvector. Large eigenvalue = important direction. Sum of eigenvalues = total variance. Eigenvalue / sum (eigenvalues) = proportion of variance explained. PCA rotates coordinate system to align with data's main variation. PC1 points in direction of maximum variance, PC2 in direction of maximum remaining variance orthogonal to PC1, etc.`,
    keyPoints: [
      'Covariance matrix captures feature relationships',
      'Eigenvectors are new axes (principal components)',
      'Eigenvalues represent variance along each axis',
      'Components ordered by decreasing eigenvalue',
      'Projects data onto directions of maximum variance',
    ],
  },
  {
    id: 'principal-component-analysis-q2',
    question: `How do you choose the number of principal components to retain? Explain the explained variance ratio and compare different selection methods (elbow, 95% threshold, cross-validation).`,
    hint: 'Cover cumulative explained variance and trade-offs between information retention and dimensionality reduction.',
    sampleAnswer: `EXPLAINED VARIANCE RATIO: Each PC's eigenvalue / total variance. Indicates importance. Cumulative sum shows total variance retained. SELECTION METHODS: (1) THRESHOLD: Keep PCs explaining 95% (or 99%) of variance. Most common. Ensures minimal information loss. (2) ELBOW METHOD: Plot explained variance vs component number. Keep components before elbow (where variance drops sharply). (3) KAISER CRITERION: Keep PCs with eigenvalue > 1 (for standardized data). Indicates PC captures more variance than single original feature. (4) CROSS-VALIDATION: Try different k, evaluate on downstream task (classification, regression). Choose k maximizing task performance. (5) INTERPRETATION: Keep components you can interpret based on loadings. TRADE-OFFS: More PCs = more information but less reduction. Fewer PCs = more reduction but information loss. PRACTICAL: Start with 95% threshold. For visualization, use 2-3 PCs regardless of variance. For preprocessing, use CV on downstream task. RULE: Always check cumulative variance plot - shows diminishing returns of additional PCs.`,
    keyPoints: [
      'Explained variance ratio = eigenvalue / total variance',
      'Common: retain PCs explaining 95-99% variance',
      'Elbow method looks for drop-off point',
      'Cross-validation chooses based on task performance',
      'Trade-off between information retention and dimensionality',
    ],
  },
  {
    id: 'principal-component-analysis-q3',
    question: `Explain how to interpret principal components through loadings. How do loadings help understand what each PC represents, and why is this important for feature engineering?`,
    hint: 'Cover what loadings are, how to create biplots, and examples of interpretation.',
    sampleAnswer: `LOADINGS: Correlations between original features and PCs. High loading = feature strongly contributes to PC. CALCULATION: Loading_ij = eigenvector_ij × sqrt (eigenvalue_j). Shows how much original feature i contributes to PC j. INTERPRETATION: PC1 loadings show which features vary together most. Positive loading = feature increases with PC. Negative loading = feature decreases with PC. EXAMPLE: If PC1 has high positive loadings for [height, weight, shoe_size], PC1 represents 'body size'. If PC2 has positive loading for age, negative for elasticity, PC2 represents 'aging'. BIPLOT: Plots data points in PC space AND loading vectors. Arrows show original features. Long arrow = feature important for those PCs. Parallel arrows = correlated features. IMPORTANCE FOR FEATURE ENGINEERING: (1) Identify correlated feature groups, (2) Understand data structure, (3) Create interpretable combinations, (4) Identify redundant features (high loading on same PC), (5) Domain validation: do PCs make sense? WARNING: PCs are linear combinations - harder to interpret than original features. Trade-off: reduced dimensionality vs interpretability.`,
    keyPoints: [
      'Loadings show correlation between features and PCs',
      'Help interpret what each PC represents',
      'Biplot visualizes both data and loadings',
      'Reveals correlated features and redundancy',
      'Trade-off: dimensionality reduction vs interpretability',
    ],
  },
];
