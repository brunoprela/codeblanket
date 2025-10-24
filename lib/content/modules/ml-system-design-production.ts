/**
 * Module: ML System Design & Production
 * Module 16 of ML/AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { mlSystemDesignPrinciples } from '../sections/ml-system-design-production/ml-system-design-principles';
import { dataEngineeringForMl } from '../sections/ml-system-design-production/data-engineering-for-ml';
import { experimentTrackingManagement } from '../sections/ml-system-design-production/experiment-tracking-management';
import { modelTrainingPipeline } from '../sections/ml-system-design-production/model-training-pipeline';
import { modelServingDeployment } from '../sections/ml-system-design-production/model-serving-deployment';
import { modelMonitoring } from '../sections/ml-system-design-production/model-monitoring';
import { abTestingForMl } from '../sections/ml-system-design-production/ab-testing-for-ml';
import { scalabilityPerformance } from '../sections/ml-system-design-production/scalability-performance';
import { automlNeuralArchitectureSearch } from '../sections/ml-system-design-production/automl-neural-architecture-search';
import { mlSecurityPrivacy } from '../sections/ml-system-design-production/ml-security-privacy';
import { mlSystemCaseStudies } from '../sections/ml-system-design-production/ml-system-case-studies';
import { mlOpsBestPractices } from '../sections/ml-system-design-production/mlops-best-practices';
import { realTimeMlSystems } from '../sections/ml-system-design-production/real-time-ml-systems';
import { llmProductionSystems } from '../sections/ml-system-design-production/llm-production-systems';

// Quiz imports
import { mlSystemDesignPrinciplesQuiz } from '../quizzes/ml-system-design-production/ml-system-design-principles';
import { dataEngineeringForMlQuiz } from '../quizzes/ml-system-design-production/data-engineering-for-ml';
import { experimentTrackingManagementQuiz } from '../quizzes/ml-system-design-production/experiment-tracking-management';
import { modelTrainingPipelineQuiz } from '../quizzes/ml-system-design-production/model-training-pipeline';
import { modelServingDeploymentQuiz } from '../quizzes/ml-system-design-production/model-serving-deployment';
import { modelMonitoringQuiz } from '../quizzes/ml-system-design-production/model-monitoring';
import { abTestingForMlQuiz } from '../quizzes/ml-system-design-production/ab-testing-for-ml';
import { scalabilityPerformanceQuiz } from '../quizzes/ml-system-design-production/scalability-performance';
import { automlNeuralArchitectureSearchQuiz } from '../quizzes/ml-system-design-production/automl-neural-architecture-search';
import { mlSecurityPrivacyQuiz } from '../quizzes/ml-system-design-production/ml-security-privacy';
import { mlSystemCaseStudiesQuiz } from '../quizzes/ml-system-design-production/ml-system-case-studies';
import { mlOpsBestPracticesQuiz } from '../quizzes/ml-system-design-production/mlops-best-practices';
import { realTimeMlSystemsQuiz } from '../quizzes/ml-system-design-production/real-time-ml-systems';
import { llmProductionSystemsQuiz } from '../quizzes/ml-system-design-production/llm-production-systems';

// Multiple choice imports
import { mlSystemDesignPrinciplesQuestions } from '../multiple-choice/ml-system-design-production/ml-system-design-principles';
import { dataEngineeringForMlQuestions } from '../multiple-choice/ml-system-design-production/data-engineering-for-ml';
import { experimentTrackingManagementQuestions } from '../multiple-choice/ml-system-design-production/experiment-tracking-management';
import { modelTrainingPipelineQuestions } from '../multiple-choice/ml-system-design-production/model-training-pipeline';
import { modelServingDeploymentQuestions } from '../multiple-choice/ml-system-design-production/model-serving-deployment';
import { modelMonitoringQuestions } from '../multiple-choice/ml-system-design-production/model-monitoring';
import { abTestingForMlQuestions } from '../multiple-choice/ml-system-design-production/ab-testing-for-ml';
import { scalabilityPerformanceQuestions } from '../multiple-choice/ml-system-design-production/scalability-performance';
import { automlNeuralArchitectureSearchQuestions } from '../multiple-choice/ml-system-design-production/automl-neural-architecture-search';
import { mlSecurityPrivacyQuestions } from '../multiple-choice/ml-system-design-production/ml-security-privacy';
import { mlSystemCaseStudiesQuestions } from '../multiple-choice/ml-system-design-production/ml-system-case-studies';
import { mlOpsBestPracticesQuestions } from '../multiple-choice/ml-system-design-production/mlops-best-practices';
import { realTimeMlSystemsQuestions } from '../multiple-choice/ml-system-design-production/real-time-ml-systems';
import { llmProductionSystemsQuestions } from '../multiple-choice/ml-system-design-production/llm-production-systems';

export const mlSystemDesignProductionModule: Module = {
  id: 'ml-system-design-production',
  title: 'ML System Design & Production',
  description:
    'Master the complete ML production lifecycle from system design to deployment and monitoring. Learn to build scalable, reliable, and maintainable ML systems including data pipelines, experiment tracking, model training infrastructure, serving architectures, monitoring systems, and MLOps best practices. Cover real-time ML systems, AutoML, security, privacy, and production LLM systems with comprehensive case studies.',
  icon: '🏗️',
  keyTakeaways: [
    'Design ML systems balancing complexity, latency, throughput, and cost constraints',
    'Build robust data pipelines: ETL/ELT, versioning, feature stores, quality monitoring',
    'Track experiments systematically with MLflow, Weights & Biases, or Neptune.ai',
    'Implement production training pipelines: distributed training, GPU optimization, hyperparameter tuning',
    'Deploy models efficiently: batch vs. real-time inference, REST APIs, containerization',
    'Monitor production models: prediction drift, data drift, performance degradation, alerting',
    'Run rigorous A/B tests: statistical significance, power analysis, multi-armed bandits',
    'Optimize for scale: model compression, quantization, caching, load balancing',
    'Apply AutoML and NAS: Auto-sklearn, H2O AutoML, automated feature engineering',
    'Secure ML systems: adversarial robustness, differential privacy, federated learning',
    'Study real systems: recommenders, search ranking, fraud detection, real-time bidding',
    'Follow MLOps best practices: CI/CD, testing strategies, reproducibility, technical debt',
    'Build real-time ML: online learning, streaming pipelines, low-latency inference',
    'Deploy LLMs in production: token streaming, cost optimization, fallback strategies',
  ],
  sections: [
    {
      ...mlSystemDesignPrinciples,
      quiz: mlSystemDesignPrinciplesQuiz,
      multipleChoice: mlSystemDesignPrinciplesQuestions,
    },
    {
      ...dataEngineeringForMl,
      quiz: dataEngineeringForMlQuiz,
      multipleChoice: dataEngineeringForMlQuestions,
    },
    {
      ...experimentTrackingManagement,
      quiz: experimentTrackingManagementQuiz,
      multipleChoice: experimentTrackingManagementQuestions,
    },
    {
      ...modelTrainingPipeline,
      quiz: modelTrainingPipelineQuiz,
      multipleChoice: modelTrainingPipelineQuestions,
    },
    {
      ...modelServingDeployment,
      quiz: modelServingDeploymentQuiz,
      multipleChoice: modelServingDeploymentQuestions,
    },
    {
      ...modelMonitoring,
      quiz: modelMonitoringQuiz,
      multipleChoice: modelMonitoringQuestions,
    },
    {
      ...abTestingForMl,
      quiz: abTestingForMlQuiz,
      multipleChoice: abTestingForMlQuestions,
    },
    {
      ...scalabilityPerformance,
      quiz: scalabilityPerformanceQuiz,
      multipleChoice: scalabilityPerformanceQuestions,
    },
    {
      ...automlNeuralArchitectureSearch,
      quiz: automlNeuralArchitectureSearchQuiz,
      multipleChoice: automlNeuralArchitectureSearchQuestions,
    },
    {
      ...mlSecurityPrivacy,
      quiz: mlSecurityPrivacyQuiz,
      multipleChoice: mlSecurityPrivacyQuestions,
    },
    {
      ...mlSystemCaseStudies,
      quiz: mlSystemCaseStudiesQuiz,
      multipleChoice: mlSystemCaseStudiesQuestions,
    },
    {
      ...mlOpsBestPractices,
      quiz: mlOpsBestPracticesQuiz,
      multipleChoice: mlOpsBestPracticesQuestions,
    },
    {
      ...realTimeMlSystems,
      quiz: realTimeMlSystemsQuiz,
      multipleChoice: realTimeMlSystemsQuestions,
    },
    {
      ...llmProductionSystems,
      quiz: llmProductionSystemsQuiz,
      multipleChoice: llmProductionSystemsQuestions,
    },
  ],
};
