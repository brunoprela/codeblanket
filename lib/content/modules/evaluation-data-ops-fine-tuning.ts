/**
 * Module: Evaluation, Data Ops & Fine-Tuning
 * Module 16 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { aiEvaluationFundamentals } from '../sections/evaluation-data-ops-fine-tuning/ai-evaluation-fundamentals';
import { llmOutputEvaluation } from '../sections/evaluation-data-ops-fine-tuning/llm-output-evaluation';
import { promptEvaluationABTesting } from '../sections/evaluation-data-ops-fine-tuning/prompt-evaluation-ab-testing';
import { evaluationDatasetsBenchmarks } from '../sections/evaluation-data-ops-fine-tuning/evaluation-datasets-benchmarks';
import { humanEvaluationFeedback } from '../sections/evaluation-data-ops-fine-tuning/human-evaluation-feedback';
import { dataLabelingAnnotation } from '../sections/evaluation-data-ops-fine-tuning/data-labeling-annotation';
import { syntheticDataGeneration } from '../sections/evaluation-data-ops-fine-tuning/synthetic-data-generation';
import { modelFineTuningFundamentals } from '../sections/evaluation-data-ops-fine-tuning/model-fine-tuning-fundamentals';
import { fineTuningOpenAIModels } from '../sections/evaluation-data-ops-fine-tuning/fine-tuning-openai-models';
import { fineTuningOpenSourceModels } from '../sections/evaluation-data-ops-fine-tuning/fine-tuning-open-source-models';
import { retrievalEvaluationRAG } from '../sections/evaluation-data-ops-fine-tuning/retrieval-evaluation-rag';
import { multiModalEvaluation } from '../sections/evaluation-data-ops-fine-tuning/multi-modal-evaluation';
import { continuousEvaluationMonitoring } from '../sections/evaluation-data-ops-fine-tuning/continuous-evaluation-monitoring';
import { buildingEvaluationPlatform } from '../sections/evaluation-data-ops-fine-tuning/building-evaluation-platform';

// Quiz imports
import { aiEvaluationFundamentalsQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/ai-evaluation-fundamentals';
import { llmOutputEvaluationQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/llm-output-evaluation';
import { promptEvaluationABTestingQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/prompt-evaluation-ab-testing';
import { evaluationDatasetsBenchmarksQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/evaluation-datasets-benchmarks';
import { humanEvaluationFeedbackQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/human-evaluation-feedback';
import { dataLabelingAnnotationQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/data-labeling-annotation';
import { syntheticDataGenerationQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/synthetic-data-generation';
import { modelFineTuningFundamentalsQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/model-fine-tuning-fundamentals';
import { fineTuningOpenAIModelsQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/fine-tuning-openai-models';
import { fineTuningOpenSourceModelsQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/fine-tuning-open-source-models';
import { retrievalEvaluationRAGQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/retrieval-evaluation-rag';
import { multiModalEvaluationQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/multi-modal-evaluation';
import { continuousEvaluationMonitoringQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/continuous-evaluation-monitoring';
import { buildingEvaluationPlatformQuiz } from '../quizzes/evaluation-data-ops-fine-tuning/building-evaluation-platform';

// Multiple choice imports
import { aiEvaluationFundamentalsMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/ai-evaluation-fundamentals';
import { llmOutputEvaluationMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/llm-output-evaluation';
import { promptEvaluationABTestingMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/prompt-evaluation-ab-testing';
import { evaluationDatasetsBenchmarksMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/evaluation-datasets-benchmarks';
import { humanEvaluationFeedbackMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/human-evaluation-feedback';
import { dataLabelingAnnotationMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/data-labeling-annotation';
import { syntheticDataGenerationMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/synthetic-data-generation';
import { modelFineTuningFundamentalsMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/model-fine-tuning-fundamentals';
import { fineTuningOpenAIModelsMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/fine-tuning-openai-models';
import { fineTuningOpenSourceModelsMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/fine-tuning-open-source-models';
import { retrievalEvaluationRAGMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/retrieval-evaluation-rag';
import { multiModalEvaluationMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/multi-modal-evaluation';
import { continuousEvaluationMonitoringMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/continuous-evaluation-monitoring';
import { buildingEvaluationPlatformMultipleChoice } from '../multiple-choice/evaluation-data-ops-fine-tuning/building-evaluation-platform';

export const evaluationDataOpsFinetuningModule: Module = {
  id: 'applied-ai-evaluation-dataops-finetuning',
  title: 'Evaluation, Data Ops & Fine-Tuning',
  description:
    'Master evaluating AI systems, managing datasets, and fine-tuning models. Build comprehensive evaluation systems, understand the complete data lifecycle, implement human-in-the-loop workflows, and fine-tune both proprietary and open-source models. Learn to create synthetic data, evaluate RAG systems, monitor production models, and build internal evaluation platforms.',
  icon: '📊',
  keyTakeaways: [
    'Design comprehensive evaluation frameworks for AI systems',
    'Implement automated LLM evaluation with metrics and guardrails',
    'Build A/B testing systems for prompts and model comparisons',
    'Create and manage high-quality evaluation datasets and benchmarks',
    'Design human evaluation protocols with inter-annotator agreement',
    'Build cost-effective data labeling pipelines with quality control',
    'Generate synthetic data with LLMs while maintaining quality',
    'Fine-tune models with LoRA, QLoRA, and full parameter updates',
    'Deploy fine-tuned OpenAI models and manage version lifecycle',
    'Fine-tune open-source models (Llama, Mistral) with memory optimization',
    'Evaluate retrieval systems (RAG) with precision, recall, and faithfulness',
    'Assess multi-modal models (vision, video, document understanding)',
    'Implement continuous evaluation and monitoring for production systems',
    'Build internal evaluation platforms with CI/CD integration',
    'Detect and diagnose model degradation with automated alerting',
    'Optimize evaluation infrastructure for performance and cost',
    'Design data pipelines with versioning and quality gates',
  ],
  sections: [
    {
      ...aiEvaluationFundamentals,
      quiz: aiEvaluationFundamentalsQuiz,
      multipleChoice: aiEvaluationFundamentalsMultipleChoice,
    },
    {
      ...llmOutputEvaluation,
      quiz: llmOutputEvaluationQuiz,
      multipleChoice: llmOutputEvaluationMultipleChoice,
    },
    {
      ...promptEvaluationABTesting,
      quiz: promptEvaluationABTestingQuiz,
      multipleChoice: promptEvaluationABTestingMultipleChoice,
    },
    {
      ...evaluationDatasetsBenchmarks,
      quiz: evaluationDatasetsBenchmarksQuiz,
      multipleChoice: evaluationDatasetsBenchmarksMultipleChoice,
    },
    {
      ...humanEvaluationFeedback,
      quiz: humanEvaluationFeedbackQuiz,
      multipleChoice: humanEvaluationFeedbackMultipleChoice,
    },
    {
      ...dataLabelingAnnotation,
      quiz: dataLabelingAnnotationQuiz,
      multipleChoice: dataLabelingAnnotationMultipleChoice,
    },
    {
      ...syntheticDataGeneration,
      quiz: syntheticDataGenerationQuiz,
      multipleChoice: syntheticDataGenerationMultipleChoice,
    },
    {
      ...modelFineTuningFundamentals,
      quiz: modelFineTuningFundamentalsQuiz,
      multipleChoice: modelFineTuningFundamentalsMultipleChoice,
    },
    {
      ...fineTuningOpenAIModels,
      quiz: fineTuningOpenAIModelsQuiz,
      multipleChoice: fineTuningOpenAIModelsMultipleChoice,
    },
    {
      ...fineTuningOpenSourceModels,
      quiz: fineTuningOpenSourceModelsQuiz,
      multipleChoice: fineTuningOpenSourceModelsMultipleChoice,
    },
    {
      ...retrievalEvaluationRAG,
      quiz: retrievalEvaluationRAGQuiz,
      multipleChoice: retrievalEvaluationRAGMultipleChoice,
    },
    {
      ...multiModalEvaluation,
      quiz: multiModalEvaluationQuiz,
      multipleChoice: multiModalEvaluationMultipleChoice,
    },
    {
      ...continuousEvaluationMonitoring,
      quiz: continuousEvaluationMonitoringQuiz,
      multipleChoice: continuousEvaluationMonitoringMultipleChoice,
    },
    {
      ...buildingEvaluationPlatform,
      quiz: buildingEvaluationPlatformQuiz,
      multipleChoice: buildingEvaluationPlatformMultipleChoice,
    },
  ],
};
