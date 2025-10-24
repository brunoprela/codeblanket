import { MultipleChoiceQuestion } from '../../../types';

export const puttingItAllTogetherMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-piat-mc-1',
    question:
      'For an AI document Q&A MVP (3 months, $50k budget), what should you intentionally skip?',
    options: [
      'Core Q&A functionality',
      'Teams/enterprise features, mobile app, API, video/audio docs, admin dashboard',
      'User authentication',
      'Payment processing',
    ],
    correctAnswer: 1,
    explanation:
      'MVP scope: Build ONLY: PDF upload, text extraction, Q&A (RAG), basic UI, auth, payment. Skip: (1) Teams/enterprise (focus on individuals first), (2) Mobile app (responsive web sufficient), (3) API (not core value), (4) Complex formats (video/audio - niche), (5) Admin dashboard (use Retool). Reasoning: Each adds 2-4 weeks development. Focus on core value, validate, then expand. Most failed startups die from trying to build everything, not from building too little.',
  },
  {
    id: 'bcap-piat-mc-2',
    question:
      'Your AI product has 1000 users and $10k MRR. What is the highest-leverage next feature?',
    options: [
      'New video support (new market)',
      'Teams/enterprise (10x revenue per customer, upsell existing users)',
      'Redesign UI',
      'Add 10 small features',
    ],
    correctAnswer: 1,
    explanation:
      'Teams/enterprise is highest leverage: (1) 10x revenue: $20/mo → $200-2000/mo, (2) Upsell existing users (no new acquisition cost), (3) Lower churn (teams stickier than individuals), (4) Proven model (Slack, Notion, Figma scaled this way), (5) Users already asking for it (validated demand). Video support: Risky (new market, different persona, unsure demand). UI redesign: Low impact on revenue. Teams delivers fastest revenue growth ($10k → $30k MRR in 2 months realistic).',
  },
  {
    id: 'bcap-piat-mc-3',
    question:
      "At $100k MRR with 2 engineers, you're overwhelmed. What should you prioritize first?",
    options: [
      'Build new features to attract more users',
      'Stop new features, fix critical bugs, hire support ($3k/mo), hire engineer ($12k/mo)',
      'Sell the company',
      'Keep working 100hr weeks',
    ],
    correctAnswer: 1,
    explanation:
      'Triage and stabilize: (1) Stop all new features (focus on stability), (2) Fix critical bugs only (data loss, outages), (3) Hire support ($3k/mo contract) - saves 50 emails/day, (4) Hire senior backend engineer ($12k/mo) - splits load, (5) Process: Intercom for support, alternating stability/feature weeks. Burn: $15k/mo, 3 months runway acceptable at $100k MRR. New features while system unstable = more bugs, more churn. Stabilize first, then grow. Most scale failures come from premature growth before infrastructure ready.',
  },
  {
    id: 'bcap-piat-mc-4',
    question:
      'What tech stack should a solo founder choose for an AI product MVP?',
    options: [
      'Microservices architecture with Kubernetes',
      'FastAPI (backend), Next.js (frontend), Claude (LLM), Qdrant (vectors), S3 (storage) - managed services',
      'Custom-built infrastructure from scratch',
      'No code tools only',
    ],
    correctAnswer: 1,
    explanation:
      'Solo founder stack: (1) FastAPI: Fast Python development, async support, good docs, (2) Next.js: React framework, API routes, easy deployment (Vercel), (3) Claude: Best reasoning, no fine-tuning needed, (4) Qdrant Cloud: Managed vector DB, (5) S3: Cheap storage. Why: (1) Proven stack (works), (2) Managed services (avoid ops burden), (3) Fast development (ship in weeks not months), (4) Scales later (can handle 10k users). Avoid: Complex architecture (Kubernetes - overkill), custom infra (maintenance burden), no-code (limited flexibility).',
  },
  {
    id: 'bcap-piat-mc-5',
    question:
      'What are realistic metrics for Month 1 after launching an AI product?',
    options: [
      '10,000 users, $100k MRR',
      '100 signups, 5 paid ($100 MRR), <1% error rate',
      '1 million users',
      'No metrics needed',
    ],
    correctAnswer: 1,
    explanation:
      'Realistic Month 1 (Product Hunt launch): (1) 100-500 signups (good launch), (2) 5-50 paying customers ($100-1000 MRR), (3) 3-5% free-to-paid conversion, (4) <1% error rate (stability), (5) >40% week-1 retention. Unrealistic expectations (10k users, $100k) cause founders to give up prematurely. Most successful products: Month 1 <$1k MRR, Month 6 $5-10k MRR, Year 1 $50-100k MRR. Focus: Product quality, user feedback, building distribution, not vanity metrics.',
  },
];
