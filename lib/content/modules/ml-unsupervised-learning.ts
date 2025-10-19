/**
 * Module: Classical Machine Learning - Unsupervised Learning
 * Topic: ML/AI
 *
 * Comprehensive coverage of clustering, dimensionality reduction, anomaly detection, and association rules
 */

import { Module } from '../../types';

// Import sections
import { unsupervisedLearningOverview } from '../sections/ml-unsupervised-learning/unsupervised-learning-overview';
import { kMeansClustering } from '../sections/ml-unsupervised-learning/k-means-clustering';
import { hierarchicalClustering } from '../sections/ml-unsupervised-learning/hierarchical-clustering';
import { dbscanDensityClustering } from '../sections/ml-unsupervised-learning/dbscan-density-clustering';
import { principalComponentAnalysis } from '../sections/ml-unsupervised-learning/principal-component-analysis';
import { otherDimensionalityReduction } from '../sections/ml-unsupervised-learning/other-dimensionality-reduction';
import { anomalyDetection } from '../sections/ml-unsupervised-learning/anomaly-detection';
import { associationRuleLearning } from '../sections/ml-unsupervised-learning/association-rule-learning';

// Import quizzes (discussion questions)
import { unsupervised_learning_overviewQuiz } from '../quizzes/ml-unsupervised-learning/unsupervised-learning-overview-quiz';
import { k_means_clusteringQuiz } from '../quizzes/ml-unsupervised-learning/k-means-clustering-quiz';
import { hierarchical_clusteringQuiz } from '../quizzes/ml-unsupervised-learning/hierarchical-clustering-quiz';
import { dbscan_density_clusteringQuiz } from '../quizzes/ml-unsupervised-learning/dbscan-density-clustering-quiz';
import { principal_component_analysisQuiz } from '../quizzes/ml-unsupervised-learning/principal-component-analysis-quiz';
import { other_dimensionality_reductionQuiz } from '../quizzes/ml-unsupervised-learning/other-dimensionality-reduction-quiz';
import { anomaly_detectionQuiz } from '../quizzes/ml-unsupervised-learning/anomaly-detection-quiz';
import { association_rule_learningQuiz } from '../quizzes/ml-unsupervised-learning/association-rule-learning-quiz';

// Import multiple choice questions
import { unsupervised_learning_overviewMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/unsupervised-learning-overview-multiple-choice';
import { k_means_clusteringMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/k-means-clustering-multiple-choice';
import { hierarchical_clusteringMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/hierarchical-clustering-multiple-choice';
import { dbscan_density_clusteringMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/dbscan-density-clustering-multiple-choice';
import { principal_component_analysisMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/principal-component-analysis-multiple-choice';
import { other_dimensionality_reductionMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/other-dimensionality-reduction-multiple-choice';
import { anomaly_detectionMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/anomaly-detection-multiple-choice';
import { association_rule_learningMultipleChoice } from '../multiple-choice/ml-unsupervised-learning/association-rule-learning-multiple-choice';

export const mlUnsupervisedLearning: Module = {
  id: 'ml-unsupervised-learning',
  title: 'Classical Machine Learning - Unsupervised Learning',
  description:
    'Master unsupervised learning techniques including clustering, dimensionality reduction, anomaly detection, and pattern discovery for unlabeled data',
  category: 'undefined',
  difficulty: 'intermediate',
  estimatedTime: 'undefined',
  prerequisites: [
    'Module 6: Python for Data Science',
    'Module 5: Statistics Fundamentals',
  ],
  icon: '🔍',
  keyTakeaways: [
    'Unsupervised learning discovers patterns in unlabeled data',
    'K-Means minimizes WCSS to create spherical clusters',
    'K-Means++ initialization improves convergence and results',
    'Hierarchical clustering creates dendrograms showing relationships at all scales',
    'Ward linkage minimizes variance like K-Means objective',
    'DBSCAN finds arbitrarily shaped clusters and identifies outliers',
    'DBSCAN uses epsilon and MinPts to define density',
    'PCA finds orthogonal axes that maximize variance',
    'Always scale features before PCA or distance-based methods',
    'Explained variance ratio determines how many components to retain',
    't-SNE preserves local structure; great for visualization',
    'UMAP is faster than t-SNE and preserves global structure',
    'Cluster sizes and inter-cluster distances in t-SNE/UMAP plots are not meaningful',
    'Isolation Forest identifies anomalies as easier-to-isolate points',
    'LOF detects local outliers based on density differences',
    'Association rules discover patterns with support, confidence, and lift metrics',
    'Apriori property enables efficient pruning of candidate itemsets',
    'Lift adjusts for base rate; more informative than confidence alone',
  ],
  learningObjectives: [
    'Understand when to use unsupervised vs supervised learning',
    'Apply K-Means clustering with appropriate K selection methods',
    'Implement hierarchical clustering and interpret dendrograms',
    'Use DBSCAN for density-based clustering and outlier detection',
    'Perform PCA for dimensionality reduction and visualization',
    'Apply t-SNE and UMAP for non-linear dimensionality reduction',
    'Detect anomalies using Isolation Forest, LOF, and statistical methods',
    'Mine association rules for market basket analysis',
    'Evaluate clustering quality with silhouette scores and domain knowledge',
    'Choose appropriate algorithms based on data characteristics and requirements',
  ],
  sections: [
    {
      ...unsupervisedLearningOverview,
      quiz: unsupervised_learning_overviewQuiz,
      multipleChoice: unsupervised_learning_overviewMultipleChoice,
    },
    {
      ...kMeansClustering,
      quiz: k_means_clusteringQuiz,
      multipleChoice: k_means_clusteringMultipleChoice,
    },
    {
      ...hierarchicalClustering,
      quiz: hierarchical_clusteringQuiz,
      multipleChoice: hierarchical_clusteringMultipleChoice,
    },
    {
      ...dbscanDensityClustering,
      quiz: dbscan_density_clusteringQuiz,
      multipleChoice: dbscan_density_clusteringMultipleChoice,
    },
    {
      ...principalComponentAnalysis,
      quiz: principal_component_analysisQuiz,
      multipleChoice: principal_component_analysisMultipleChoice,
    },
    {
      ...otherDimensionalityReduction,
      quiz: other_dimensionality_reductionQuiz,
      multipleChoice: other_dimensionality_reductionMultipleChoice,
    },
    {
      ...anomalyDetection,
      quiz: anomaly_detectionQuiz,
      multipleChoice: anomaly_detectionMultipleChoice,
    },
    {
      ...associationRuleLearning,
      quiz: association_rule_learningQuiz,
      multipleChoice: association_rule_learningMultipleChoice,
    },
  ],
};
