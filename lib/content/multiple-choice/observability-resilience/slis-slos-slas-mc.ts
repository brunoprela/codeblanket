/**
 * Multiple choice questions for SLIs, SLOs, and SLAs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const slisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the relationship between SLI, SLO, and SLA?',
    options: [
      'They are all the same thing',
      'SLI is what you measure, SLO is your target, SLA is the customer promise',
      'SLI is for customers, SLO is internal, SLA is for metrics',
      'SLI is optional, SLO and SLA are required',
    ],
    correctAnswer: 1,
    explanation:
      'The hierarchy is: SLI (Service Level Indicator) = what you measure (e.g., "% of successful requests"). SLO (Service Level Objective) = internal target for SLI (e.g., "99.9% success rate"). SLA (Service Level Agreement) = customer-facing promise with financial consequences (e.g., "99.9% or receive service credit"). SLO should be stricter than SLA to provide buffer. Example: SLA 99.9%, SLO 99.95% (internal target with 0.05% safety margin).',
  },
  {
    id: 'mc2',
    question: 'What is an error budget?',
    options: [
      'The total number of errors allowed',
      'The amount of unreliability allowed by the SLO (100% - SLO%)',
      'The budget for fixing errors',
      'The number of engineers assigned to fix errors',
    ],
    correctAnswer: 1,
    explanation:
      "Error budget = 100% - SLO%. It's the amount of unreliability allowed by your SLO. Example: SLO 99.9% → Error budget 0.1% → 43.2 minutes downtime/month allowed. Decision framework: Budget remaining → ship features. Budget near 0 → slow releases, focus reliability. Budget exhausted → feature freeze, only reliability work. This quantifies the trade-off between innovation and stability.",
  },
  {
    id: 'mc3',
    question: 'How much downtime per month does a 99.9% SLO allow?',
    options: ['4.32 minutes', '43.2 minutes', '7.2 hours', '3.65 days'],
    correctAnswer: 1,
    explanation:
      '99.9% SLO allows 0.1% downtime. Calculation: 30 days × 24 hours × 60 minutes = 43,200 minutes/month. 43,200 × 0.1% = 43.2 minutes downtime allowed. Common SLO downtimes: 90% = 3 days/month, 99% = 7.2 hours/month, 99.9% = 43.2 minutes/month, 99.99% = 4.32 minutes/month, 99.999% = 26 seconds/month. Each additional nine becomes exponentially more expensive.',
  },
  {
    id: 'mc4',
    question: 'Why should your SLO be stricter than your SLA?',
    options: [
      'To confuse customers',
      'To provide a safety buffer so you can breach SLO without breaching SLA',
      'SLO should always equal SLA',
      'To make engineering harder',
    ],
    correctAnswer: 1,
    explanation:
      'SLO should be stricter than SLA to provide a safety buffer. Example: SLA 99.9% (customer promise), SLO 99.95% (internal target), Buffer 0.05%. If you breach your internal SLO (99.95%), you still have buffer before breaching customer SLA (99.9%) and incurring financial penalties. This gives you time to fix issues before customers are affected. Without buffer, any SLO breach immediately triggers SLA penalties.',
  },
  {
    id: 'mc5',
    question: 'What is SLO-based alerting (error budget burn rate)?',
    options: [
      'Alerting when SLO is breached',
      'Alerting when error budget is consumed too fast (predicting SLO breach)',
      'Alerting on all errors',
      'Disabling alerts based on SLO',
    ],
    correctAnswer: 1,
    explanation:
      "SLO-based alerting alerts on error budget burn rate (how fast you're consuming your monthly budget), not when SLO is breached (too late!). Example: Monthly budget 43.2 minutes. Fast burn alert: If burning 2% of budget in 1 hour → At this rate, budget exhausted in 2 days → Page immediately. Medium burn alert: If burning 5% in 6 hours → Budget exhausted in 5 days → Create ticket. This enables proactive response before SLO breach instead of reactive response after.",
  },
];
