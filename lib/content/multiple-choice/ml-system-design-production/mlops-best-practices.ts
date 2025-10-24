import { MultipleChoiceQuestion } from '@/lib/types';

export const mlOpsBestPracticesQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'mobp-mc-1',
    question:
      "Your ML team has deployed 15 models in production. Each model's training pipeline is manually triggered with custom scripts, making it difficult to reproduce results and track model versions. What is the most critical first step to improve this situation?",
    options: [
      'Implement comprehensive model monitoring for all 15 models',
      'Establish CI/CD pipelines with automated testing and version control for training code and models',
      'Migrate all models to a unified serving infrastructure',
      "Create detailed documentation for each model's training process",
    ],
    correctAnswer: 1,
    explanation:
      "CI/CD with version control (code, data, models) is foundational for MLOps: it enables reproducibility, automated testing, and systematic model management. This addresses the core issues (reproducibility, tracking). Once established, monitoring, unified serving, and documentation become easier to implement systematically. Option A (monitoring) is important but doesn't solve reproducibility. Option C (unified serving) helps operations but not training issues. Option D (documentation) is valuable but doesn't automate or enforce good practices like CI/CD does.",
    difficulty: 'intermediate',
    topic: 'MLOps Best Practices',
  },
  {
    id: 'mobp-mc-2',
    question:
      "You're implementing automated testing for an ML training pipeline. Which testing strategy provides the most comprehensive coverage for catching issues before production?",
    options: [
      'Unit tests for data processing functions only',
      'Multi-layered testing: unit tests (functions), integration tests (pipeline), model tests (performance), data tests (quality)',
      'End-to-end tests that run the full training pipeline on a sample dataset',
      'Manual testing with human review before each deployment',
    ],
    correctAnswer: 1,
    explanation:
      "Comprehensive multi-layered testing catches different types of issues: unit tests (function correctness), integration tests (component interactions), model tests (performance regression, bias), data tests (schema validation, distribution checks). This is the ML adaptation of the test pyramid. Option A (unit tests only) misses integration and model-specific issues. Option C (only E2E) is slow and doesn't localize bugs well. Option D (manual testing) doesn't scale and is error-prone.",
    difficulty: 'advanced',
    topic: 'MLOps Best Practices',
  },
  {
    id: 'mobp-mc-3',
    question:
      'Your team maintains multiple models that share common preprocessing logic. This logic is duplicated across model repositories, leading to inconsistencies. What is the best practice to address this?',
    options: [
      'Create a shared preprocessing library with versioning, testing, and documented releases',
      'Copy the preprocessing code to each model repository to maintain independence',
      'Use a monorepo to store all models and share code directly',
      'Document the preprocessing logic and trust developers to implement it consistently',
    ],
    correctAnswer: 0,
    explanation:
      "Shared library with versioning is best practice: it ensures consistency, allows independent updates with version pinning, and provides a single point for testing and documentation. Models can depend on specific versions for stability. Option B (duplication) causes inconsistencies and maintenance burden. Option C (monorepo) creates tight coupling—changes affect all models simultaneously. Option D (documentation only) is error-prone and doesn't prevent drift.",
    difficulty: 'intermediate',
    topic: 'MLOps Best Practices',
  },
  {
    id: 'mobp-mc-4',
    question:
      'Your model retraining pipeline runs weekly. Occasionally, retrained models perform worse than the current production model. What automated safeguard would best prevent deploying degraded models?',
    options: [
      'Implement holdout test set validation with automated performance comparison against production model',
      'Always deploy the newly trained model to maintain freshness',
      'Have a human manually review all model metrics before deployment',
      'Use A/B testing after deployment to catch issues',
    ],
    correctAnswer: 0,
    explanation:
      "Automated validation on a holdout test set with performance comparison (gating mechanism) prevents deploying degraded models. Only deploy if new model meets minimum absolute performance AND relative improvement criteria (or non-degradation). This catches issues before production impact. Option B (always deploy) risks degradation. Option C (manual review) doesn't scale and is error-prone. Option D (A/B testing) catches issues but only after exposing users to the bad model.",
    difficulty: 'advanced',
    topic: 'MLOps Best Practices',
  },
  {
    id: 'mobp-mc-5',
    question:
      'Your ML system has accumulated technical debt: models use deprecated features, training code is poorly documented, and deployment is manual. Management wants to focus on new models. What should you prioritize?',
    options: [
      'Pause new model development and fix all technical debt first',
      'Continue with new models and ignore technical debt until it causes production issues',
      'Allocate time systematically: dedicate 20-30% of sprint to technical debt reduction while building new models',
      'Rewrite the entire ML platform from scratch',
    ],
    correctAnswer: 2,
    explanation:
      'Systematic debt allocation (e.g., 20-30% of capacity) balances business needs (new models) with sustainability (reducing debt). This prevents debt from compounding while maintaining feature velocity. Address highest-impact debt first. Option A (pause everything) may not be business-viable. Option B (ignore debt) leads to eventual crisis and inability to move quickly. Option D (full rewrite) is risky, expensive, and may introduce new issues—incremental improvement is safer.',
    difficulty: 'advanced',
    topic: 'MLOps Best Practices',
  },
];
