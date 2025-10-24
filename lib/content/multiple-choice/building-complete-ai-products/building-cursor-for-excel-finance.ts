/**
 * Multiple choice questions for Building Cursor for Excel & Finance
 */

import { MultipleChoiceQuestion } from '../../../types';

export const buildingCursorForExcelFinanceMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-building-cursor-for-excel-finance-mc-1',
      question:
        'What is the most critical consideration when building building cursor for excel & finance?',
      options: [
        'Using the latest technology stack',
        'User experience and reliability',
        'Minimizing development time',
        'Maximizing feature count',
      ],
      correctAnswer: 1,
      explanation:
        'User experience and reliability are paramount. Users need a system that works consistently and provides value. Technology choices should serve UX, not vice versa. Shipping reliable software that solves real problems beats having the latest tech stack.',
    },
    {
      id: 'bcap-building-cursor-for-excel-finance-mc-2',
      question:
        'How should you approach scaling building cursor for excel & finance?',
      options: [
        'Scale vertically only',
        'Design for horizontal scaling from the start',
        'Wait until performance issues arise',
        'Always use microservices',
      ],
      correctAnswer: 1,
      explanation:
        'Design for horizontal scaling from the start by making services stateless, using external session storage, and avoiding instance-specific data. This allows adding more instances as demand grows. Vertical scaling has limits, and retrofitting for scale is expensive.',
    },
    {
      id: 'bcap-building-cursor-for-excel-finance-mc-3',
      question:
        'What is the best strategy for managing costs in building cursor for excel & finance?',
      options: [
        'Always use the cheapest option',
        'Track metrics, optimize based on data, balance cost vs quality',
        'Ignore costs until profitable',
        'Only use open-source tools',
      ],
      correctAnswer: 1,
      explanation:
        'Effective cost management requires tracking metrics (cost per user, cost per request), identifying optimization opportunities through data, and making informed trade-offs between cost and quality. Blindly choosing cheapest options sacrifices quality; ignoring costs leads to unsustainable unit economics.',
    },
    {
      id: 'bcap-building-cursor-for-excel-finance-mc-4',
      question:
        'How should errors be handled in production building cursor for excel & finance?',
      options: [
        'Show generic error messages to users',
        'Retry failed operations, implement fallbacks, communicate clearly with users',
        'Let errors crash the application',
        "Log errors but don't notify users",
      ],
      correctAnswer: 1,
      explanation:
        'Production error handling requires: automatic retries for transient failures, fallback options (alternative models, cached responses), clear user communication about what happened and what they can do, comprehensive logging for debugging. This maintains user trust and system reliability.',
    },
    {
      id: 'bcap-building-cursor-for-excel-finance-mc-5',
      question:
        'What monitoring is essential for building cursor for excel & finance?',
      options: [
        'Only error logs',
        'Performance metrics, error tracking, user analytics, cost metrics',
        'Just uptime monitoring',
        'No monitoring needed',
      ],
      correctAnswer: 1,
      explanation:
        'Comprehensive monitoring includes: performance (latency, throughput), errors (rate, types, traces), user behavior (feature usage, conversion), and costs (per user, per request). This data drives optimization decisions and alerts teams to issues before users complain. You cannot improve what you do not measure.',
    },
  ];
