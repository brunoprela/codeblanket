export default {
  id: 'fin-m15-s11-quiz',
  title: 'Margin and Collateral Management - Quiz',
  questions: [
    {
      id: 1,
      question:
        'What is the key difference between initial margin and variation margin?',
      options: [
        'Initial margin is posted at trade inception; variation margin is posted daily',
        'Initial margin covers potential future exposure; variation margin settles current exposure',
        'Initial margin is segregated; variation margin may be reused',
        'All of the above',
      ],
      correctAnswer: 3,
      explanation:
        "All three statements are correct and capture different aspects of the IM/VM distinction. Initial margin is posted upfront to cover potential future moves (what if counterparty defaults tomorrow? IM provides cushion during close-out period). Variation margin settles mark-to-market changes daily (current exposure). IM is segregated (belongs to poster, held by custodian) while VM transfers ownership (can be reused by receiver). Option A: True—timing difference. Option B: True—purpose difference. Option C: True—legal treatment difference. Post-UMR (Uncleared Margin Rules), bilateral derivatives require IM+VM just like cleared derivatives. The $100B+ cost of UMR comes from IM being trapped (can't be reused), whereas pre-UMR many derivatives had only VM or no margin at all.",
    },
    {
      id: 2,
      question:
        'An ISDA CSA specifies: Threshold = $10M, MTA = $500K. Current exposure is $10.3M, collateral posted is $10M. What happens?',
      options: [
        'Margin call for $300K',
        'No margin call (below MTA)',
        'Margin call for $500K',
        'Return $500K collateral to posting party',
      ],
      correctAnswer: 1,
      explanation:
        'Required collateral = Exposure - Threshold = $10.3M - $10M = $300K. But margin call amount ($300K) is below MTA ($500K), so NO call is made. MTA exists to reduce operational burden—avoid tiny margin calls. Collateral stays at $10M until gap exceeds MTA. If exposure rises to $10.6M, gap = $600K > $500K MTA, then call is made for $600K (not $500K—must cover full gap once MTA breached). Option A ignores MTA. Option C would call for MTA amount (wrong—call for actual gap if >MTA). Option D is backwards. MTA tradeoff: Lower MTA = better risk management (more frequent collateral updates) but higher operational cost. Typical MTAs: $100K-$500K for institutional relationships. Zero MTA = call for any gap (operationally intensive but best risk management).',
    },
    {
      id: 3,
      question:
        'Why is the cheapest-to-deliver problem important in collateral optimization?',
      options: [
        'Posting expensive collateral (cash at 3% funding cost) when cheap collateral (Treasuries at 0.5% repo rate) is acceptable wastes money',
        'Some collateral is not accepted by counterparties',
        'Collateral valuation changes over time',
        'Collateral must be delivered physically',
      ],
      correctAnswer: 0,
      explanation:
        'Collateral optimization saves millions by posting cheapest acceptable collateral. If CSA accepts cash OR Treasuries, and cash costs 3% to fund while Treasuries cost 0.5% (repo rate), posting Treasuries saves 2.5% annually. On $1B collateral, this is $25M/year savings! Option B is true but not the core of cheapest-to-deliver. Option C is true but not the main issue. Option D is wrong—mostly electronic. The optimization problem: Given margin call of $100M and inventory of (Cash, Treasuries, Agencies, Corporates), which combination to post? Subject to: CSA eligibility, haircuts (may need $102M gross to deliver $100M net), concentrations limits, future flexibility. Large firms have sophisticated optimization systems running daily. Without optimization, firms leave $10M-50M/year on table by posting suboptimal collateral.',
    },
    {
      id: 4,
      question:
        "Post-UMR, a firm's initial margin requirements for bilateral derivatives increased from $0 to $500M. What is the main economic impact?",
      options: [
        'The firm must post $500M that cannot be reused (funding cost)',
        "The firm's counterparties are safer",
        'Derivatives trading becomes more expensive',
        'All of the above',
      ],
      correctAnswer: 3,
      explanation:
        'All three are correct. UMR forces $500M IM posting that is segregated (can\'t be reused), creating funding cost of ~1-3% = $5M-15M/year (option A). Counterparties are safer because if firm defaults, they have $500M cushion to replace trades (option B—the intended benefit). This cost must be passed to clients, making derivatives more expensive, reducing trading volume (option C—unintended consequence). The $500M is "trapped liquidity"—must fund it but can\'t use it. MVA (Margin Valuation Adjustment) captures this cost—for a 10-year derivative requiring $10M IM, MVA could be $5M+ (50% of IM value). This has reduced derivatives activity 20-30% post-UMR as the all-in cost (bid-ask + XVA + MVA) makes some trades uneconomical. Regulators accept this as price of safer system post-2008.',
    },
    {
      id: 5,
      question:
        'A firm receives margin call for $50M. It can post: (1) Cash (0% haircut, 3% funding cost), (2) Treasuries (2% haircut, 0.8% repo cost), (3) Agencies (4% haircut, 1.5% repo cost). Which should it post?',
      options: [
        'Cash—no haircut',
        'Treasuries—lowest all-in cost',
        'Agencies—highest yield',
        'Diversify across all three',
      ],
      correctAnswer: 1,
      explanation:
        "Calculate all-in cost including haircuts: (1) Cash: Post $50M, cost = $50M × 3% = $1.5M/year. (2) Treasuries: Post $51.02M (50/(1-0.02)), cost = $51.02M × 0.8% = $408K/year. (3) Agencies: Post $52.08M (50/(1-0.04)), cost = $52.08M × 1.5% = $781K/year. Treasuries win: $408K vs $1.5M cash vs $781K agencies. Post Treasuries! Option A (cash) is expensive. Option C is wrong—agencies have higher yield on assets you own, but here you're borrowing them (pay repo rate). Option D (diversify) doesn't minimize cost. This is why firms hold Treasury inventories—they're the cheapest collateral to post. Large firms optimize daily across 1000+ relationships, saving $10M-50M/year vs. naively posting cash everywhere. Key insight: Haircut matters but funding cost matters more—2% haircut on 0.8% asset beats 0% haircut on 3% asset.",
    },
  ],
} as const;
