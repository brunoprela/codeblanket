/**
 * Quiz: Anomaly Detection
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const anomaly_detectionQuiz: QuizQuestion[] = [
  {
    id: 'anomaly-detection-q1',
    question: `Compare statistical methods (Z-score, IQR) with machine learning methods (Isolation Forest, LOF) for anomaly detection. What are the assumptions, advantages, and limitations of each approach?`,
    hint: 'Consider distributional assumptions, multivariate vs univariate, and scalability.',
    sampleAnswer: `STATISTICAL METHODS: Z-SCORE: Assumes normal distribution. Identifies points > 3 standard deviations from mean. Pros: simple, interpretable, fast. Cons: assumes normality, univariate (checks each feature independently), misses multivariate outliers (normal on each feature but unusual combination). IQR: Uses quartiles (Q1, Q3). Outliers outside [Q1-1.5×IQR, Q3+1.5×IQR]. Pros: robust to distribution, doesn't assume normality. Cons: still univariate, arbitrary threshold. ML METHODS: ISOLATION FOREST: Tree-based. Anomalies are easier to isolate (fewer splits). Pros: handles multivariate, no distribution assumption, fast, works in high-D. Cons: parameters (contamination) hard to set. LOF (Local Outlier Factor): Density-based. Compares point's density to neighbors'. Pros: finds local anomalies, handles varying density. Cons: slow (O(n²)), sensitive to parameters. WHEN TO USE: Statistical: quick baseline, univariate, normal data. Isolation Forest: general-purpose, high-D, large datasets. LOF: local anomalies, varying density, smaller datasets. BEST PRACTICE: Start with Isolation Forest for most cases. Combine methods for robustness.`,
    keyPoints: [
      'Statistical: simple, univariate, assumes distribution',
      'Z-score assumes normality; IQR more robust',
      'Isolation Forest: multivariate, no assumptions, scalable',
      'LOF: local anomalies, handles varying density, slower',
      'Isolation Forest best for general-purpose use',
    ],
  },
  {
    id: 'anomaly-detection-q2',
    question: `Explain the Isolation Forest algorithm. Why are anomalies easier to isolate than normal points, and how does this translate to shorter path lengths in isolation trees?`,
    hint: 'Cover the intuition behind isolation, tree construction, and path length scoring.',
    sampleAnswer: `CORE INTUITION: Anomalies are 'few and different' - they're rare and far from normal points. This makes them easier to separate (isolate) from the rest of the data. ALGORITHM: (1) Build many random trees (forest). Each tree: recursively split data on random features at random thresholds until points isolated. (2) Record path length for each point: number of splits from root to leaf. (3) Anomaly score: based on average path length across all trees. Short path = anomaly. WHY ANOMALIES HAVE SHORT PATHS: Consider splitting randomly: NORMAL POINT in dense region requires many splits to separate from neighbors (long path). ANOMALY in sparse region gets isolated quickly with few splits (short path). ANALOGY: Finding a lone person in empty field (easy, few steps) vs finding specific person in crowd (hard, many steps). MATHEMATICAL: Expected path length E(h (x)) for normal points ≈ average binary tree height ≈ 2(log n - 1). Anomalies have h (x) << log n. ANOMALY SCORE: Normalized: s (x) = 2^(-E(h (x))/c) where c normalizes. s ≈ 1: anomaly, s ≈ 0.5: normal, s < 0.5: safe.`,
    keyPoints: [
      'Anomalies are few and different, easier to isolate',
      'Random trees: split on random features and values',
      'Path length: splits needed to isolate point',
      'Normal points in dense regions: long paths',
      'Anomalies in sparse regions: short paths',
    ],
  },
  {
    id: 'anomaly-detection-q3',
    question: `In anomaly detection, what is the contamination parameter and how do you set it? What happens if you set it incorrectly, and how can you validate your anomaly detection results?`,
    hint: 'Cover the precision-recall trade-off and validation strategies when labels are unavailable.',
    sampleAnswer: `CONTAMINATION: Expected proportion of anomalies in dataset. Sets threshold for classification. contamination=0.1 means expect 10% anomalies. SETTING IT: (1) DOMAIN KNOWLEDGE: often best source. Fraud rate? Defect rate? (2) DATA EXPLORATION: plot anomaly scores, look for natural threshold. (3) BUSINESS REQUIREMENTS: cost of false positives vs false negatives. (4) START: 1-5% if unknown. INCORRECT SETTING: TOO HIGH (e.g., 0.2 when true 0.01): many false positives, normal points flagged, wastes investigation time. TOO LOW (e.g., 0.01 when true 0.1): miss many anomalies, false negatives, fail to catch issues. VALIDATION WITHOUT LABELS: (1) DOMAIN EXPERTS: manually review flagged anomalies. Do they make sense? (2) ANOMALY SCORE DISTRIBUTION: check for clear separation. (3) CONSISTENCY: do multiple methods agree? (4) TEMPORAL: do anomalies repeat or are they one-time events? (5) IMPACT: if acted upon, did flagged anomalies cause real problems? WITH LABELS (rare): use precision, recall, F1, PR curve. BEST PRACTICE: Set conservatively (low contamination), review flagged cases, adjust based on precision of actual anomalies.`,
    keyPoints: [
      'Contamination: expected proportion of anomalies',
      'Use domain knowledge or start with 1-5%',
      'Too high: many false positives; too low: miss anomalies',
      'Validate with domain experts and score distributions',
      'Monitor and adjust based on real-world performance',
    ],
  },
];
