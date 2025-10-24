/**
 * Module: Scaling & Cost Optimization
 * Module 13 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { horizontalScaling } from '../sections/scaling-cost-optimization/horizontal-scaling';
import { modelSelectionRouting } from '../sections/scaling-cost-optimization/model-selection-routing';
import { promptOptimizationForCost } from '../sections/scaling-cost-optimization/prompt-optimization-for-cost';
import { cachingAtScale } from '../sections/scaling-cost-optimization/caching-at-scale';
import { batchProcessing } from '../sections/scaling-cost-optimization/batch-processing';
import { edgeComputingForLlms } from '../sections/scaling-cost-optimization/edge-computing-for-llms';
import { databaseScaling } from '../sections/scaling-cost-optimization/database-scaling';
import { quantizationModelOptimization } from '../sections/scaling-cost-optimization/quantization-model-optimization';
import { multiRegionDeployment } from '../sections/scaling-cost-optimization/multi-region-deployment';
import { loadTestingCapacityPlanning } from '../sections/scaling-cost-optimization/load-testing-capacity-planning';
import { costMonitoringAnalysis } from '../sections/scaling-cost-optimization/cost-monitoring-analysis';
import { buildingScalingStrategy } from '../sections/scaling-cost-optimization/building-scaling-strategy';

// Quiz imports
import { horizontal_scalingQuiz } from '../quizzes/scaling-cost-optimization/horizontal-scaling';
import { model_selection_routingQuiz } from '../quizzes/scaling-cost-optimization/model-selection-routing';
import { prompt_optimization_for_costQuiz } from '../quizzes/scaling-cost-optimization/prompt-optimization-for-cost';
import { caching_at_scaleQuiz } from '../quizzes/scaling-cost-optimization/caching-at-scale';
import { batch_processingQuiz } from '../quizzes/scaling-cost-optimization/batch-processing';
import { edge_computing_for_llmsQuiz } from '../quizzes/scaling-cost-optimization/edge-computing-for-llms';
import { database_scalingQuiz } from '../quizzes/scaling-cost-optimization/database-scaling';
import { quantization_model_optimizationQuiz } from '../quizzes/scaling-cost-optimization/quantization-model-optimization';
import { multi_region_deploymentQuiz } from '../quizzes/scaling-cost-optimization/multi-region-deployment';
import { load_testing_capacity_planningQuiz } from '../quizzes/scaling-cost-optimization/load-testing-capacity-planning';
import { cost_monitoring_analysisQuiz } from '../quizzes/scaling-cost-optimization/cost-monitoring-analysis';
import { building_scaling_strategyQuiz } from '../quizzes/scaling-cost-optimization/building-scaling-strategy';

// Multiple choice imports
import { horizontal_scalingMC } from '../multiple-choice/scaling-cost-optimization/horizontal-scaling';
import { model_selection_routingMC } from '../multiple-choice/scaling-cost-optimization/model-selection-routing';
import { prompt_optimization_for_costMC } from '../multiple-choice/scaling-cost-optimization/prompt-optimization-for-cost';
import { caching_at_scaleMC } from '../multiple-choice/scaling-cost-optimization/caching-at-scale';
import { batch_processingMC } from '../multiple-choice/scaling-cost-optimization/batch-processing';
import { edge_computing_for_llmsMC } from '../multiple-choice/scaling-cost-optimization/edge-computing-for-llms';
import { database_scalingMC } from '../multiple-choice/scaling-cost-optimization/database-scaling';
import { quantization_model_optimizationMC } from '../multiple-choice/scaling-cost-optimization/quantization-model-optimization';
import { multi_region_deploymentMC } from '../multiple-choice/scaling-cost-optimization/multi-region-deployment';
import { load_testing_capacity_planningMC } from '../multiple-choice/scaling-cost-optimization/load-testing-capacity-planning';
import { cost_monitoring_analysisMC } from '../multiple-choice/scaling-cost-optimization/cost-monitoring-analysis';
import { building_scaling_strategyMC } from '../multiple-choice/scaling-cost-optimization/building-scaling-strategy';

export const scalingCostOptimizationModule: Module = {
  id: 'scaling-cost-optimization',
  title: 'Scaling & Cost Optimization',
  description:
    'Master scaling LLM applications from 10 to 10 million users while optimizing costs by 50-90%. Learn horizontal scaling, model routing, caching strategies, and build production-ready systems that handle massive scale efficiently.',
  icon: '💰',
  keyTakeaways: [
    'Horizontal scaling enables unlimited growth by adding more servers',
    'Model routing can reduce costs by 50-90% using cheaper models appropriately',
    'Prompt optimization reduces token usage and costs significantly',
    'Multi-layer caching (L1/L2/semantic) achieves 60-95% cost reduction',
    'Batch processing improves efficiency for non-real-time workloads',
    'Edge deployment reduces latency by 50-90% for global users',
    'Database scaling requires read replicas, connection pooling, and sharding',
    'Quantization enables running large models on smaller hardware',
    'Multi-region deployment improves availability and reduces latency',
    'Load testing identifies bottlenecks before they impact users',
    'Cost monitoring and analysis enables data-driven optimization',
    'Comprehensive scaling strategy plans for 10x, 100x, 1000x growth',
  ],
  learningObjectives: [
    'Design and implement horizontally scaled LLM architectures',
    'Build intelligent model routing systems that optimize costs',
    'Optimize prompts to reduce token usage by 50-80%',
    'Implement production-grade caching with Redis and semantic caching',
    'Build batch processing systems for efficient background work',
    'Deploy LLM applications at the edge with Cloudflare/Vercel',
    'Scale databases to handle millions of users',
    'Apply quantization to reduce model size and inference costs',
    'Deploy multi-region architectures with automatic failover',
    'Conduct load testing and capacity planning',
    'Build comprehensive cost monitoring and analysis systems',
    'Create scaling roadmaps that handle 1000x growth',
  ],
  prerequisites: [
    'Module 1: LLM Engineering Fundamentals',
    'Module 12: Production LLM Applications',
    'Understanding of APIs and backend development',
    'Basic knowledge of databases and caching',
  ],
  sections: [
    {
      id: 'horizontal-scaling',
      title: 'Horizontal Scaling',
      content: horizontalScaling.content,
      multipleChoiceQuestions: horizontal_scalingMC.questions,
      discussionQuestions: horizontal_scalingQuiz.questions,
    },
    {
      id: 'model-selection-routing',
      title: 'Model Selection & Routing',
      content: modelSelectionRouting.content,
      multipleChoiceQuestions: model_selection_routingMC.questions,
      discussionQuestions: model_selection_routingQuiz.questions,
    },
    {
      id: 'prompt-optimization-for-cost',
      title: 'Prompt Optimization for Cost',
      content: promptOptimizationForCost.content,
      multipleChoiceQuestions: prompt_optimization_for_costMC.questions,
      discussionQuestions: prompt_optimization_for_costQuiz.questions,
    },
    {
      id: 'caching-at-scale',
      title: 'Caching at Scale',
      content: cachingAtScale.content,
      multipleChoiceQuestions: caching_at_scaleMC.questions,
      discussionQuestions: caching_at_scaleQuiz.questions,
    },
    {
      id: 'batch-processing',
      title: 'Batch Processing',
      content: batchProcessing.content,
      multipleChoiceQuestions: batch_processingMC.questions,
      discussionQuestions: batch_processingQuiz.questions,
    },
    {
      id: 'edge-computing-for-llms',
      title: 'Edge Computing for LLMs',
      content: edgeComputingForLlms.content,
      multipleChoiceQuestions: edge_computing_for_llmsMC.questions,
      discussionQuestions: edge_computing_for_llmsQuiz.questions,
    },
    {
      id: 'database-scaling',
      title: 'Database Scaling',
      content: databaseScaling.content,
      multipleChoiceQuestions: database_scalingMC.questions,
      discussionQuestions: database_scalingQuiz.questions,
    },
    {
      id: 'quantization-model-optimization',
      title: 'Quantization & Model Optimization',
      content: quantizationModelOptimization.content,
      multipleChoiceQuestions: quantization_model_optimizationMC.questions,
      discussionQuestions: quantization_model_optimizationQuiz.questions,
    },
    {
      id: 'multi-region-deployment',
      title: 'Multi-Region Deployment',
      content: multiRegionDeployment.content,
      multipleChoiceQuestions: multi_region_deploymentMC.questions,
      discussionQuestions: multi_region_deploymentQuiz.questions,
    },
    {
      id: 'load-testing-capacity-planning',
      title: 'Load Testing & Capacity Planning',
      content: loadTestingCapacityPlanning.content,
      multipleChoiceQuestions: load_testing_capacity_planningMC.questions,
      discussionQuestions: load_testing_capacity_planningQuiz.questions,
    },
    {
      id: 'cost-monitoring-analysis',
      title: 'Cost Monitoring & Analysis',
      content: costMonitoringAnalysis.content,
      multipleChoiceQuestions: cost_monitoring_analysisMC.questions,
      discussionQuestions: cost_monitoring_analysisQuiz.questions,
    },
    {
      id: 'building-scaling-strategy',
      title: 'Building a Scaling Strategy',
      content: buildingScalingStrategy.content,
      multipleChoiceQuestions: building_scaling_strategyMC.questions,
      discussionQuestions: building_scaling_strategyQuiz.questions,
    },
  ],
  practicalProjects: [
    {
      title: 'Horizontally Scaled Chat Service',
      description:
        'Build a stateless chat service that scales across multiple servers with Redis for shared state',
      difficulty: 'Advanced',
      estimatedTime: '4-6 hours',
      skills: ['Load balancing', 'Stateless design', 'Redis', 'Auto-scaling'],
    },
    {
      title: 'Multi-Model Router',
      description:
        'Implement intelligent routing between GPT-4, GPT-3.5, and Claude based on query complexity',
      difficulty: 'Advanced',
      estimatedTime: '6-8 hours',
      skills: [
        'Model routing',
        'Complexity classification',
        'Cost optimization',
      ],
    },
    {
      title: 'Production Caching System',
      description:
        'Build multi-layer cache (L1/L2/semantic) that reduces costs by 80%+',
      difficulty: 'Advanced',
      estimatedTime: '8-12 hours',
      skills: ['Redis', 'Semantic caching', 'Cache invalidation'],
    },
    {
      title: 'Batch Processing Pipeline',
      description:
        'Create batch processing system for 100K daily tasks with priority queues',
      difficulty: 'Advanced',
      estimatedTime: '6-10 hours',
      skills: ['Celery', 'Priority queues', 'Job scheduling'],
    },
    {
      title: 'Cost Monitoring Dashboard',
      description:
        'Build real-time cost tracking with anomaly detection and budget enforcement',
      difficulty: 'Advanced',
      estimatedTime: '8-12 hours',
      skills: ['Cost tracking', 'Anomaly detection', 'Real-time dashboards'],
    },
    {
      title: 'Complete Scaling Strategy',
      description:
        'Design comprehensive scaling plan from 1K to 1M users with cost projections',
      difficulty: 'Expert',
      estimatedTime: '12-16 hours',
      skills: ['Capacity planning', 'Architecture design', 'Cost modeling'],
    },
  ],
  estimatedTime: '2-3 weeks',
  difficulty: 'Advanced',
};
