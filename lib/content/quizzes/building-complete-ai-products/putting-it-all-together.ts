export const puttingItAllTogetherQuiz = [
  {
    id: 'bcap-piat-q-1',
    question:
      'You\'re building "DocuMind" - an AI document Q&A platform - from scratch. You have 3 months and $50k budget (your own money). Design the complete plan: (1) MVP feature scope, (2) Tech stack, (3) Development timeline, (4) Launch strategy, (5) Success metrics. What do you build first, second, third? What do you intentionally skip? How do you know if it\'s working?',
    sampleAnswer:
      'Month 1 - MVP core: (1) Features: PDF upload, text extraction, Q&A (RAG), basic UI. Skip: DOCX/images, chat history, teams, analytics. (2) Tech stack: Backend (FastAPI + Anthropic Claude), Frontend (Next.js), DB (PostgreSQL), Vector (Qdrant Cloud), Storage (S3). Why: Fast development, proven stack, managed services (avoid ops). (3) Week 1-2: Backend - upload PDF, extract text (PyPDF2), chunk, embed (Voyage), store in Qdrant. (4) Week 3: RAG - retrieve relevant chunks, build prompt, call Claude, stream response. (5) Week 4: Frontend - upload UI, chat interface, SSE streaming. Deploy on Vercel (frontend) + Render (backend). Total cost: $200/mo. Month 2 - Polish + launch: (1) Week 5: Improve extraction (handle scanned PDFs with Tesseract). (2) Week 6: Add auth (Clerk), payment (Stripe), rate limiting. Pricing: $20/mo for 100 docs. (3) Week 7: Polish UI, add examples, write docs. (4) Week 8: Launch - Product Hunt, HN, Twitter. Goal: 100 signups, 5 paid. Budget: $10k for ads/promotion. Month 3 - Iterate: (1) Week 9-10: Add most-requested features (chat history, doc management). (2) Week 11: Second launch (v2.0), target: 500 users, 25 paid ($500 MRR). (3) Week 12: Outbound sales (cold email), partnerships (integrate with Notion). Skip intentionally: Teams/enterprise (focus on individuals), mobile app (use responsive web), API (not core), video/audio docs (niche), admin dashboard (use Retool). Success metrics: Month 1: MVP deployed, 10 beta users giving feedback. Month 2: 100 signups, 5 paid ($100 MRR), <1% error rate. Month 3: 500 users, 25 paid ($500 MRR), 3% conversion, >40% W1 retention. If not hitting targets by Month 2, pivot or shutdown (burn rate: $15k/month).',
    keyPoints: [
      'Month 1: MVP core only (PDF + Q&A), skip everything else',
      'Tech stack: FastAPI, Next.js, Claude, Qdrant, S3 (managed services)',
      'Month 2: Polish, launch (Product Hunt/HN), goal 100 signups',
      'Month 3: Iterate based on feedback, outbound sales, 500 users goal',
      'Skip: teams, mobile, API, complex formats (focus on core value)',
    ],
  },
  {
    id: 'bcap-piat-q-2',
    question:
      'DocuMind has 1000 users and $10k MRR after 6 months. You have 3 growth options: (1) Add video support (opens new market), (2) Build teams/enterprise (10x revenue per customer), (3) Launch API (developer ecosystem). Each takes 2 months development. Which do you choose? How do you validate before building? What are the risks?',
    sampleAnswer:
      'Analysis: (1) Video support: New market (video creators, educators), but: different user persona, competitive (Descript exists), high cost (video storage/processing), unsure demand. (2) Teams/enterprise: 10x revenue ($200/mo → $2000/mo), existing users want it (upsell), proven model (Notion, Figma), requires: permissioning, SSO, admin dashboard. (3) API: Developer ecosystem, viral growth (devs build on platform), usage-based revenue, but: commoditization risk, support burden, unclear revenue. Validation before building: (1) Video: Survey users "Would you use video transcription?" (>40% yes = proceed). Interview 10 users, ask: "What video tools do you use? Why?" Check: Market size (YouTube creators?), willingness to pay. (2) Teams: Email current users: "We\'re building team features (shared docs, collaboration). Interested? What\'s your budget?" Pre-sell: "Pay $500/mo now, get early access." If 10 commit = $5k MRR, validated. (3) API: Create waitlist, post on HN, see sign-ups. Offer: Free beta for first 100 devs. If >200 sign up in 1 week, validated. Recommendation: Choose teams/enterprise. Reasoning: (1) Fastest revenue: Upsell existing users (no new acquisition). (2) Proven model: Slack, Notion, Figma all scaled this way. (3) Lower risk: Users already asking for it. (4) Compounds: Teams have lower churn (multiple users = stickier). Implementation: Month 1: Build permissioning (roles: admin/member), shared workspaces. Month 2: SSO, admin dashboard, billing for teams. Launch with 10 design partners ($2k/mo each = $20k MRR). Risks: (1) Over-engineering: Keep MVP simple, add enterprise features later. (2) Sales complexity: Teams want demos/trials. Hire SDR. (3) Support: Teams need more handholding. Video/API: Defer until $50k MRR (established, can diversify).',
    keyPoints: [
      'Validate before building: surveys, interviews, pre-sales',
      'Teams/enterprise best: upsell existing users, 10x revenue, proven model',
      'Video: new market (risky), high cost, unsure demand',
      'API: viral potential, but commoditization risk, unclear revenue',
      'Recommendation: teams (fastest revenue, lowest risk, compounds)',
    ],
  },
  {
    id: 'bcap-piat-q-3',
    question:
      "DocuMind is at $100k MRR, 5k users, 2 engineers (you + co-founder). You're overwhelmed: bugs piling up, support emails (50/day), feature requests (100+), infrastructure issues (outages). What do you do? Prioritize: (1) Hire (who first?), (2) Process (support, engineering), (3) Infrastructure (stability), (4) Product (ship features vs fix bugs). You have $50k runway. What\'s the 60-day plan?",
    sampleAnswer:
      '60-day survival plan: (1) Triage (Week 1): Stop all new features. Fix critical bugs only (data loss, outages). Set expectations: Public roadmap, "We\'re fixing stability." Measure: Error rate, P95 latency, uptime. (2) Hire (Week 2-4): First hire: Customer support (contract, $3k/mo). Takes 50 emails/day off your plate. Second hire: Senior backend engineer (full-time, $12k/mo). Splits engineering load. Total burn: $15k/mo, 3 months runway. (3) Support process (Week 3): Setup: Intercom for support, Notion for FAQ, Loom videos for common issues. Goal: Support answers 80%, you handle 20%. SLA: 24hr response time. (4) Engineering process (Week 4-6): Weekly sprints, alternating: Stability week (fix bugs, improve infra), Feature week (ship 1 feature). Prioritize: High-impact, low-effort (RICE scoring). Setup: Error tracking (Sentry), monitoring (Grafana), on-call rotation. (5) Infrastructure (Week 5-8): Move to managed services: RDS (PostgreSQL), ElastiCache (Redis), reduce ops burden. Setup: Auto-scaling, load balancer, zero-downtime deploys. Budget: $5k/mo infra (acceptable at $100k MRR). (6) Product (Week 7-8): Review 100+ feature requests, group by theme. Pick top 3 themes (most requested). Build 1 per month. Communicate: "We heard you, here\'s the plan." Say no to rest. Metrics: Week 1-2: Fix 20 bugs, reduce error rate 50%. Week 3-4: Support hired, you spend 50% less time on emails. Week 5-6: Zero outages, uptime >99.9%. Week 7-8: Ship 2 high-demand features, user satisfaction up. By Day 60: Stable product, support handled, 1 engineer hired, burn controlled. Can focus on growth again.',
    keyPoints: [
      'Stop new features, focus on stability (bugs, outages)',
      'Hire: support first ($3k/mo, saves 50 emails/day), then engineer ($12k/mo)',
      'Process: Intercom for support, alternating stability/feature weeks',
      'Infrastructure: managed services (RDS, ElastiCache), auto-scaling',
      'Product: review requests, pick top 3 themes, say no to rest',
    ],
  },
];
