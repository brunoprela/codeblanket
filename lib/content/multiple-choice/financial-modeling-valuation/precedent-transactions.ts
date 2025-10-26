import { MultipleChoiceQuestion } from '@/lib/types';

export const precedentTransactionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pt-mc-1',
    question: 'A company trading at $60/share receives an acquisition offer at $80/share. What is the acquisition premium?',
    options: ['25.0%', '33.3%', '20.0%', '75.0%'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: 33.3% premium. Formula: Premium = (Offer Price - Trading Price) / Trading Price = ($80 - $60) / $60 = $20 / $60 = 0.333 = 33.3%. Common mistake: Option 1 (25%) incorrectly calculates ($80 - $60) / $80 = 25%—this is wrong denominator (should use unaffected trading price, not offer price). Option 3 (20%) is $20 premium / $80 offer, also wrong denominator. Option 4 (75%) is nonsensical. Interpretation: 33.3% premium is typical for strategic acquisition (range: 25-40%). Represents value of control, synergies, and strategic fit. If premium were only 15%, might suggest friendly deal with limited competition. If 50%+, suggests competitive bidding or hostile takeover.',
  },
  {
    id: 'pt-mc-2',
    question: 'When comparing precedent transaction multiples to trading comps multiples, which statement is most accurate?',
    options: [
      'Transaction multiples and trading multiples should be approximately equal',
      'Transaction multiples are typically 20-40% higher due to control premium and synergies',
      'Trading multiples are typically higher because public markets are more liquid',
      'The difference depends entirely on the industry with no general pattern',
    ],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Transaction multiples are 20-40% higher than trading multiples. Reason: M&A prices include (1) Control premium (25-35%): buyer controls strategy, management, board. Minority public shareholders have no control—worth less. (2) Synergies: cost savings, revenue synergies, tax benefits that buyer can realize. (3) Strategic value: eliminate competitor, acquire technology/talent, enter new market. Example: Company trades at 15x EBITDA (public market), acquires for 20x EBITDA (33% higher). Formula: Transaction Multiple ≈ Trading Multiple × (1 + Control Premium %). Option 1 is wrong—they differ systematically. Option 3 is backwards—liquidity affects trading multiples (more liquid = tighter spreads) but transaction multiples still higher. Option 4 is wrong—while magnitude varies by industry, direction is consistent (transaction > trading). If transaction multiple < trading multiple, red flag: distressed sale, fire sale, or data error.',
  },
  {
    id: 'pt-mc-3',
    question: 'A precedent transaction shows a company acquired for $2B with $100M EBITDA (20x multiple). Research reveals the buyer estimated $400M PV of synergies. What is the implied standalone valuation multiple?',
    options: ['20x EBITDA (synergies don\'t affect multiple)', '16x EBITDA (adjusting for synergies)', '24x EBITDA (including synergies)', '14x EBITDA (conservative estimate)'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: 16x EBITDA standalone multiple. Calculation: Transaction price: $2.0B. Less: PV of synergies: $0.4B (buyer-specific value). Equals: Standalone value: $1.6B. Standalone multiple: $1.6B / $100M EBITDA = 16x. Logic: Buyer paid $2B total but received $100M EBITDA asset PLUS $400M synergies. Standalone value (what asset is worth without synergies) = $2B - $400M = $1.6B. For your target company: If buyer has synergies: Can justify 20x. If buyer has NO synergies (financial buyer): Use 16x standalone. Option 1 (20x) is headline multiple but overstates standalone value. Option 3 (24x) makes no sense—would be adding synergies twice. Option 4 (14x) is arbitrary discount, no justification. Application: Always investigate whether transaction price includes significant synergies. Read proxy statements, earnings calls. If yes, back out synergies to get standalone multiple for comps analysis. Otherwise you\'re comparing synergy-inflated price to target with no synergies (apples to oranges).',
  },
  {
    id: 'pt-mc-4',
    question: 'You are using a transaction from 3 years ago in your precedent transaction analysis. The risk-free rate has increased from 2% to 5% since then. How should you adjust the historical transaction multiple?',
    options: [
      'Increase the multiple by 50% to reflect higher rates (from 2% to 5%)',
      'Decrease the multiple by approximately 20-30% to reflect higher discount rates',
      'No adjustment needed—transaction multiples are independent of interest rates',
      'Adjust only if the transaction involved debt financing',
    ],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Decrease multiple by 20-30% for 3% rate increase. Rationale: Valuation multiples inverse to discount rates. When WACC increases, present value of future cash flows decreases, lowering valuation multiples. If risk-free rate increased 3% (from 2% to 5%), WACC likely increased 2-3% (from ~8% to ~11%, assuming beta effects). Impact on perpetuity: At 8% WACC, 3% growth: 1/(0.08-0.03) = 20x. At 11% WACC, 3% growth: 1/(0.11-0.03) = 12.5x. Decline: (12.5-20)/20 = -37.5% (aggressive). Conservative adjustment: -25% to -30% is reasonable. If historical transaction was 20x EBITDA, adjust to 20x × 0.75 = 15x for current valuation. Option 1 (increase 50%) is backwards—higher rates = lower valuations, not higher. Option 3 (no adjustment) ignores time value of money—rates fundamentally drive valuations. Option 4 (only if debt financing) misunderstands—rates affect all valuations (equity discount rate increased via CAPM). Best practice: If using transaction >2 years old and rates changed significantly, either (a) exclude it (use only recent), or (b) adjust for rate environment change.',
  },
  {
    id: 'pt-mc-5',
    question: 'A company was acquired for $1.5B consisting of: $900M cash, $300M stock, and $300M earn-out (contingent on 20% revenue growth). For precedent transaction analysis, what value should you use?',
    options: [
      '$1.5B—the headline transaction value',
      '$900M—only the cash portion is certain',
      '$1.2B—cash plus stock, excluding uncertain earn-out',
      '$1.2B-$1.4B—adjust earn-out for probability of achievement',
    ],
    correctAnswer: 3,
    explanation: 'Option 4 is correct: $1.2B-$1.4B adjusted for earn-out probability. Reasoning: Components: (1) Cash ($900M): Full value (certain, immediate). (2) Stock ($300M): Full or discounted value. If lockup period, discount 10-15% for illiquidity and market risk. If no lockup and liquid, full value. Assume $300M here (no lockup mentioned). (3) Earn-out ($300M): Contingent value. Historical data: 60% of earn-outs pay 50-80% of maximum. Assume 50-70% probability of full payment. Expected value: $300M × 0.60 (midpoint) = $180M. Total expected value: $900M + $300M + $180M = $1.38B (round to $1.4B). Range: Conservative (30% earn-out probability): $900M + $300M + $90M = $1.29B. Optimistic (70% earn-out probability): $900M + $300M + $210M = $1.41B. Range: $1.2B-$1.4B. Option 1 ($1.5B headline) overstates value—includes full earn-out at face value (unrealistic). Option 2 ($900M cash only) too conservative—ignores stock and any earn-out value. Option 3 ($1.2B) ignores earn-out entirely—better than full value but still conservative. Presentation: Show both. Headline: $1.5B (press release value). Adjusted: $1.2B-$1.4B (economic value with earn-out risk). Use adjusted value for comps multiple calculation. Footnote structure in comps table.',
  },
];
