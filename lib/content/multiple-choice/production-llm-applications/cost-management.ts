import { MultipleChoiceQuestion } from '../../../types';

export const costManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-cost-mc-1',
    question: 'What is the most effective way to reduce LLM costs?',
    options: [
      'Use only free models',
      'Implement aggressive caching (60-90% cost reduction)',
      'Reduce usage',
      'Ignore costs',
    ],
    correctAnswer: 1,
    explanation:
      'Caching provides the biggest ROI: 70-90% hit rate saves 70-90% of API costs with relatively simple implementation. Far more effective than other strategies.',
  },
  {
    id: 'pllm-cost-mc-2',
    question: 'How should you track LLM costs?',
    options: [
      'Monthly bills only',
      'Record every API call with model, tokens, cost in database for granular analysis',
      'Estimates only',
      'No tracking',
    ],
    correctAnswer: 1,
    explanation:
      'Track every API call with full details (user_id, model, tokens, cost, timestamp) enabling analysis by user/model/feature and cost optimization.',
  },
  {
    id: 'pllm-cost-mc-3',
    question: 'What is model routing for cost optimization?',
    options: [
      'Load balancing',
      'Using cheaper models (GPT-3.5) when quality difference is minimal vs expensive models (GPT-4)',
      'Caching',
      'Rate limiting',
    ],
    correctAnswer: 1,
    explanation:
      'Model routing intelligently uses GPT-3.5 ($0.002) for simple tasks and GPT-4 ($0.06) only when needed, saving 97% on routed requests.',
  },
  {
    id: 'pllm-cost-mc-4',
    question: 'How do you prevent unexpected cost spikes?',
    options: [
      'Hope for the best',
      'Set budget alerts at 80%, per-user limits, monitor hourly burn rate',
      'Post-payment review',
      'No prevention possible',
    ],
    correctAnswer: 1,
    explanation:
      'Proactive prevention: budget alerts at 80%/90%, per-user daily limits, real-time burn rate monitoring, alerts on anomalous spikes, automatic limiting.',
  },
  {
    id: 'pllm-cost-mc-5',
    question: 'What ROI should caching provide?',
    options: [
      '10x',
      '2x',
      '50-100x (cache infrastructure $50/mo vs $2500/mo API savings)',
      'Break even',
    ],
    correctAnswer: 2,
    explanation:
      'With 80% hit rate on 100K requests/day: save $4800/mo on API costs vs ~$60/mo cache infrastructure = 80x ROI. Pays for implementation in weeks.',
  },
];
