export const realOptionsValuationQuiz = [
  {
    id: 'rov-q-1',
    question:
      'A biotech R&D project has traditional NPV of -$20M (PV of cash flows $80M, investment $100M today). However, you can invest $15M for Phase I trials, then CHOOSE after 1 year: invest $85M more for Phase II-III if successful, OR abandon if failed. Phase I success probability = 40%. Volatility = 60%. Risk-free rate = 4%. Should you proceed? Calculate option value.',
    sampleAnswer:
      'Real option analysis: Traditional NPV = -$20M → REJECT. But staging creates option: Phase I ($15M) buys 1-year call option. Option parameters: S = $80M (PV of cash flows if successful), K = $85M (Phase II cost), T = 1 year, σ = 60%, r = 4%. Black-Scholes call value = $28M. Decision: Pay $15M for option worth $28M → Net value = $13M (ACCEPT). Key insight: Staging converts negative NPV to positive by adding abandonment flexibility.',
    keyPoints: [
      'Staged investment creates abandonment option; Phase I trials are call option on Phase II',
      'Black-Scholes values flexibility: S = PV of success, K = future investment, σ = uncertainty',
      'High volatility (60%) increases option value—uncertainty is valuable with flexibility',
    ],
  },
  {
    id: 'rov-q-2',
    question:
      'A manufacturing plant investment has NPV = $10M. Adding expansion capacity (for $5M extra) allows doubling production if demand grows 30%+. Expansion option value = $12M (using real options model). Should you build with expansion capacity?',
    sampleAnswer:
      'Expansion option analysis: Base plant: NPV = $10M. Plant with expansion capacity: Cost = $5M extra, Option value = $12M, Net benefit = $12M - $5M = $7M. Total value with flexibility: $10M + $7M = $17M. Decision: Build with expansion capacity—increases value by 70%. Key insight: Pay $5M for growth option worth $12M.',
    keyPoints: [
      'Expansion options add value by enabling response to favorable scenarios (upside participation)',
      'Cost of flexibility ($5M) < option value ($12M) → build flexible capacity',
      'Growth options common in infrastructure, capacity planning, platform businesses',
    ],
  },
  {
    id: 'rov-q-3',
    question:
      'Traditional DCF values oil field at $200M (assume operate forever). Real options DCF values at $280M (includes abandonment if oil <$40/bbl). Explain: (a) Where does $80M extra value come from?, (b) Why traditional DCF understates value?, (c) When is difference largest?',
    sampleAnswer:
      'Abandonment option value: (a) $80M = value of stopping operations in low-price scenarios. Traditional DCF assumes operate forever (even at loss). Real options recognizes you abandon if oil <$40/bbl → avoid losses, capture only upside. (b) Traditional DCF uses expected cash flows (average of good and bad scenarios). Real options captures asymmetry: full upside when prices high, capped downside (abandon when low). (c) Difference largest when: High volatility (oil prices swing 40%+), Operating leverage (high fixed costs make losses severe), Long time horizon (more opportunities to exercise abandonment). Key: Abandonment puts floor on losses → increases value.',
    keyPoints: [
      'Abandonment option = put option on project; limits downside while preserving upside',
      'Traditional DCF assumes symmetric payoff; real options captures asymmetry from flexibility',
      'Value difference largest with high volatility, operating leverage, and long horizons',
    ],
  },
];
