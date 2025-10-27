export const sumOfPartsValuationQuiz = [
  {
    id: 'sotp-q-1',
    question:
      'A conglomerate has 3 segments: Consumer (60% revenue, 12x EV/EBITDA), Tech (30% revenue, 25x EV/EBITDA), Industrial (10% revenue, 8x EV/EBITDA). Current company trades at aggregate 15x EV/EBITDA. Calculate SOTP valuation and conglomerate discount. What drives the discount?',
    sampleAnswer:
      "SOTP vs aggregate valuation: Assume total EBITDA = $1B. Consumer: $600M EBITDA × 12x = $7.2B. Tech: $300M EBITDA × 25x = $7.5B. Industrial: $100M EBITDA × 8x = $0.8B. SOTP EV = $15.5B. Current valuation: $1B EBITDA × 15x = $15B. Conglomerate discount: ($15.5B - $15B) / $15.5B = 3.2%. Drivers: (1) Market applies average multiple (15x) instead of segment-specific. (2) Tech segment (25x) gets dragged down by lower-multiple segments. (3) Complexity discount (investors don't understand diversification). Value unlock: Spin off tech segment → trades at 25x = $7.5B standalone vs $4.5B embedded.",
    keyPoints: [
      'SOTP values each segment at appropriate multiple; aggregation masks high-multiple segments',
      'Conglomerate discount occurs when market applies average multiple vs segment-specific multiples',
      'Value unlock via spinoff: High-multiple segment (tech 25x) escapes low-multiple anchor (aggregate 15x)',
    ],
  },
  {
    id: 'sotp-q-2',
    question:
      'A company has $100M corporate costs (unallocated). In SOTP, do you: (a) Ignore (not attributable), (b) Allocate across segments, (c) Subtract as separate line item? Defend your choice.',
    sampleAnswer:
      "Corporate cost treatment: Correct answer: (c) Subtract as separate line item. Rationale: SOTP values segments as standalone businesses (excludes corporate overhead). Corporate costs ($100M/year): CEO, CFO, corporate strategy, legal, investor relations—not segment-specific. Valuation: Capitalize corporate costs at 10-12x EBITDA = $1B-$1.2B value drag. SOTP = Sum of segments - Corporate cost drag - Net debt. Why not (a) ignore: Corporate costs are real cash burn—ignoring overstates value. Why not (b) allocate: Arbitrary allocation distorts segment values. Consumer segment shouldn't bear tech CEO salary. Best practice: Show segments gross (as if standalone), then subtract corporate drag separately. Transparent, shows true cost of conglomerate structure.",
    keyPoints: [
      'Corporate costs are real cash burn; capitalize at 10-12x EBITDA as value drag ($100M × 10x = $1B drag)',
      "Don't allocate to segments (arbitrary); show as separate line item for transparency",
      'Corporate cost drag = inefficiency of conglomerate structure; eliminated in spinoff (value unlock)',
    ],
  },
  {
    id: 'sotp-q-3',
    question:
      'SOTP shows company worth $30B (sum of parts), market cap = $25B (17% discount). Board debates: Spin off high-growth tech segment or keep integrated? Analyze trade-offs.',
    sampleAnswer:
      'Spinoff vs integration analysis: Spinoff pros: (1) Unlock value: $30B SOTP vs $25B market = $5B value creation (17% uplift). (2) Pure-play premiums: Tech segment trades at growth multiple (20x+ vs 12x blended). (3) Strategic focus: Each CEO focuses on one business. (4) Capital allocation: Tech reinvests in growth, consumer returns cash. (5) Investor choice: Growth investors buy tech, value investors buy consumer. Spinoff cons: (1) Diseconomies: Lose scale in procurement, shared services. (2) Stranded costs: Corporate costs $100M must be reallocated (or cut). (3) Management distraction: 6-12 months executing spinoff. (4) Tax leakage: Some spinoffs trigger tax (though most are tax-free). Keep integrated pros: (1) Synergies: Tech serves consumer segment (internal customer). (2) Cash flow diversification: Consumer cash funds tech growth. (3) No transaction costs: Spinoff costs $50M+ (advisors, systems separation). Keep integrated cons: (1) Conglomerate discount persists: $5B value unrealized. (2) Suboptimal capital allocation: Tech starved of capital, consumer over-capitalized. Recommendation: Spin off if: (1) 17% discount > transaction costs + lost synergies, (2) No strategic synergies between segments, (3) Activist pressure or shareholder demand. Keep if: (1) Material synergies (>$500M annual), (2) Integration benefits exceed discount, (3) Strong capital allocation discipline.',
    keyPoints: [
      'Spinoff unlocks conglomerate discount (17% = $5B value) via pure-play premiums and strategic focus',
      'Trade-offs: Value unlock vs diseconomies of scale, stranded costs, and lost synergies',
      'Decision: Spin off if discount > (transaction costs + lost synergies); keep if material integration benefits',
    ],
  },
];
