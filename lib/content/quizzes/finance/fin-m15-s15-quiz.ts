export default {
  id: 'fin-m15-s15-quiz',
  title: 'BlackRock Aladdin Architecture Study - Quiz',
  questions: [
    {
      id: 1,
      question:
        'What is the primary architectural advantage of Aladdin over point solution approaches?',
      options: [
        'Aladdin is cheaper than buying multiple systems',
        'Single source of truth—portfolio management, risk, trading, operations all use same data (no reconciliation)',
        'Aladdin has better user interface',
        'Aladdin is required by regulators',
      ],
      correctAnswer: 1,
      explanation:
        "Aladdin's power is unified data architecture: All modules (PM, risk, trading, ops) reference one position database. Traditional firms have separate systems that must reconcile constantly—causes breaks, errors, delays, operational risk. With Aladdin: (1) No reconciliation needed (guaranteed consistency), (2) Real-time risk awareness (PM sees risk impact before trading), (3) Operational efficiency (18x faster workflows), (4) Single audit trail. Option A is actually wrong—Aladdin license ($30K-100K/user/year) seems expensive but TCO can be lower vs. multiple systems + integration + IT staff. Option C is subjective. Option D is wrong—not required. Example: PM makes trade → instantly see updated VaR (same system). Traditional: Trade in OMS → wait hours for risk system to update → discover breach too late. This architectural choice is why Aladdin dominates despite high price.",
    },
    {
      id: 2,
      question:
        'Aladdin manages $10T+ in assets across 1,000 firms. This creates which type of competitive advantage?',
      options: [
        'Brand recognition',
        'Network effects—data aggregation improves models for all users',
        'Regulatory approval',
        'Marketing advantage',
      ],
      correctAnswer: 1,
      explanation:
        "Network effects: With $10T assets under management, Aladdin sees aggregate supply/demand across markets, improving: Pricing (consensus valuations from 1,000 portfolios), Risk models (credit risk from massive dataset), Market intelligence (trading patterns). Each new client improves the platform for all users. Option A is true but not the competitive advantage. Option C doesn't apply. Option D is marketing, not structural advantage. Network effects create moat: New competitor starts with zero data—can't match Aladdin's insights from 30 years × 1,000 firms. This is similar to Google (more searches → better search) or Amazon (more customers → better recommendations). Network effects explain why Aladdin maintains 40-50% market share despite high pricing and why competitors struggle to gain traction even with better technology. Once a platform achieves critical mass, network effects create winner-take-most dynamics.",
    },
    {
      id: 3,
      question:
        'Why do institutional investors pay $30K-100K per user per year for Aladdin?',
      options: [
        'Aladdin has a monopoly and can charge whatever it wants',
        'Replaces multiple systems ($50M annual cost) with one ($36M), plus switching costs make alternatives expensive',
        'Institutions are price-insensitive',
        'Aladdin is required for regulatory compliance',
      ],
      correctAnswer: 1,
      explanation:
        "TCO (Total Cost of Ownership) analysis: Without Aladdin: Portfolio system $5M + Risk $8M + Trading $3M + Ops $4M + Market data $10M + IT staff $15M (50 people) + Integration $5M = $50M/year. With Aladdin: License $30M + IT staff $6M (20 people) = $36M/year + better integration + lower operational risk. Saves $14M/year! Plus switching costs ($30M-115M + 18-36 months) lock clients in. Option A is cynical—Aladdin has competitors (Bloomberg AIM, SimCorp), price reflects value. Option C is wrong—institutions are very price-sensitive. Option D is wrong—not required. The pricing works because: (1) Consolidation saves money, (2) Network effects add value, (3) Switching costs prevent churn, (4) Mission-critical (can't fail). Firms complain about price but pay because alternatives are worse (build in-house = $100M+, competitors lack features).",
    },
    {
      id: 4,
      question: "What is the biggest threat to Aladdin's competitive moat?",
      options: [
        'Open-source alternatives',
        'Cloud-native competitors with lower costs + regulators requiring open architecture due to concentration risk',
        'Better marketing by competitors',
        'Clients building in-house systems',
      ],
      correctAnswer: 1,
      explanation:
        'Two threats converging: (1) Cloud-native startups built on modern infrastructure (lower cost, faster innovation) could challenge if they achieve scale, (2) Regulators increasingly concerned about concentration risk—$10T on one platform is systemic risk; may mandate open APIs and portability to reduce lock-in. Option A is unlikely—operational risk management too complex for open-source (no successful open ERP beat SAP). Option C is irrelevant—Aladdin sells itself via pilots. Option D is rare—only largest firms attempt ($100M+ cost, 5+ years, usually fail). The regulatory threat is real: If regulators force standardized APIs and data portability, switching costs drop dramatically. Combined with cloud-native competitors having 10x lower infrastructure costs, Aladdin could face pressure. However, moat is still strong (5-10 year safety): Network effects, institutional knowledge, switching costs remain formidable even with regulatory changes.',
    },
    {
      id: 5,
      question:
        "Aladdin's risk analytics calculate VaR across millions of positions in real-time. The key technical enabler is:",
      options: [
        'Proprietary quantum computers',
        'Hierarchical aggregation + distributed computing + caching + GPU acceleration',
        'Outsourcing calculations to India',
        'Using only simple parametric VaR',
      ],
      correctAnswer: 1,
      explanation:
        'Aladdin achieves scale through multiple techniques: (1) Hierarchical aggregation—only recalculate changed sub-portfolios (100x speedup), (2) Distributed computing—1000s of servers in parallel (50x speedup), (3) Caching—reuse Greeks, covariance matrices (10x speedup), (4) GPU acceleration—Monte Carlo simulations on GPU farms (100x speedup). Combined: 10,000,000x speedup vs naive approach! Option A is science fiction. Option C is wrong—calculations run in data centers (latency matters). Option D is wrong—Aladdin uses multiple methods (parametric for real-time, Monte Carlo for EOD). The architecture allows: Sub-second VaR for 100K position portfolios, 10-second stress tests (100 scenarios), Real-time pre-trade checks (<100ms). This is why firms can\'t replicate Aladdin—requires 30 years of optimization + massive infrastructure investment. The "easy" part is models; the "hard" part is doing it at this scale with this speed.',
    },
  ],
} as const;
