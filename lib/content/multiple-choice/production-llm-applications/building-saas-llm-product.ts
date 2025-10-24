import { MultipleChoiceQuestion } from '../../../types';

export const buildingSaasLlmProductMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-saas-mc-1',
    question: 'How should you implement multi-tenant data isolation?',
    options: [
      'Trust client',
      'Separate database per tenant',
      'Single database with tenant_id filtering and Row-Level Security',
      'No isolation',
    ],
    correctAnswer: 2,
    explanation:
      'Use single database with tenant_id column, automatically filter all queries by tenant, implement Row-Level Security (RLS) policies for enforcement.',
  },
  {
    id: 'pllm-saas-mc-2',
    question: 'What payment provider should you use for SaaS subscriptions?',
    options: [
      'Build your own',
      'Stripe for subscription management, webhooks, and automatic billing',
      'Cash only',
      'Manual invoices',
    ],
    correctAnswer: 1,
    explanation:
      'Stripe handles subscriptions, automatic billing, proration, webhooks (payment success/failure), compliance, saving months of development.',
  },
  {
    id: 'pllm-saas-mc-3',
    question: 'How should you track usage for billing?',
    options: [
      'Estimates',
      'Record every request with tenant_id, aggregate daily, enforce limits',
      'Monthly manual count',
      'No tracking',
    ],
    correctAnswer: 1,
    explanation:
      'Record every API call with tenant_id, aggregate in UsageRecords table, check limits before processing, display usage dashboard per tenant.',
  },
  {
    id: 'pllm-saas-mc-4',
    question: 'What should a SaaS admin dashboard include?',
    options: [
      'Just revenue',
      'Revenue, active tenants, usage stats, top spenders, system health, support tickets',
      'Nothing',
      'Only errors',
    ],
    correctAnswer: 1,
    explanation:
      'Comprehensive dashboard: MRR/ARR, active tenants by tier, churn, usage statistics, top spenders, system health, support queue, cost analytics.',
  },
  {
    id: 'pllm-saas-mc-5',
    question: 'How do you maximize conversion in onboarding?',
    options: [
      'Require everything upfront',
      'Progressive disclosure, free trial without card, interactive demo, reduce friction',
      'No onboarding',
      'Complex forms',
    ],
    correctAnswer: 1,
    explanation:
      'Maximize conversion: free trial without requiring card, progressive disclosure (skip optional steps), interactive demo, social login, save progress.',
  },
];
