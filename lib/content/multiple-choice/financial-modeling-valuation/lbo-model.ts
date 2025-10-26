import { MultipleChoiceQuestion } from '@/lib/types';

export const lboModelMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'lbo-mc-1',
    question: 'A PE firm invests $400M equity in an LBO. After 5 years, the company is sold and equity investors receive $1.2B. What is the IRR?',
    options: ['200% (tripled investment)', '24.6%', '20.0%', '30.0%'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: 24.6% IRR. Calculation: MOIC = $1.2B / $400M = 3.0x. For 5-year hold: (1 + IRR)^5 = 3.0, IRR = 3.0^(1/5) - 1 = 3.0^0.2 - 1 = 1.246 - 1 = 0.246 = 24.6%. Common mistake: Option 1 (200%) confuses MOIC with IRR. 3x MOIC = 200% total return, but IRR accounts for time. The 200% gain occurred over 5 years, not instantly. Option 3 (20%) is close but incorrect—would require (1.20)^5 = 2.49x MOIC, not 3.0x. Option 4 (30%) is too high—would require (1.30)^5 = 3.71x MOIC. Key insight: Time matters! 3x MOIC in 3 years = 44% IRR (excellent), 3x in 5 years = 25% IRR (good), 3x in 7 years = 17% IRR (mediocre). MOIC alone is insufficient; always calculate IRR.',
  },
  {
    id: 'lbo-mc-2',
    question: 'In an LBO, which of the following contributes MOST to equity value creation over a typical 5-year hold period?',
    options: ['Multiple expansion (buying at 7x, selling at 9x EBITDA)', 'Debt paydown from cash flow generation', 'EBITDA growth from operational improvements', 'Tax shields from interest expense deductibility'],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: EBITDA growth typically contributes 40-50% of returns. Breakdown of typical LBO returns: (1) EBITDA growth: 40-50% of value creation. Growing EBITDA from $100M to $150M at 7x multiple = $350M value increase. (2) Debt paydown: 30-40% of value creation. Deleveraging from $600M to $200M debt = $400M more equity value. (3) Multiple expansion: 10-20% of value creation. Buying at 7x, selling at 8x on $150M EBITDA = $150M value increase. (4) Tax shields: ~5% benefit (present value of interest tax deduction). Option 1 (multiple expansion) is risky—can\'t control market multiples. 2021-2022 showed multiples compress quickly (rising rates). Option 2 (debt paydown) is significant (30-40%) but less than EBITDA growth. Option 4 (tax shields) provides benefit but smallest contribution (~5%). Best LBO investments: Focus on operational improvements (EBITDA growth) backed by financial engineering (debt paydown). Never underwrite multiple expansion—treat it as upside, not base case.',
  },
  {
    id: 'lbo-mc-3',
    question: 'A company has $100M EBITDA and $500M debt (5.0x leverage). It generates $60M annual FCF. If all FCF is used to pay down debt, how many years to reach 3.0x leverage (assuming EBITDA stays flat)?',
    options: ['3.3 years', '5.0 years', '2.0 years', '8.3 years'],
    correctAnswer: 0,
    explanation: 'Option 1 is correct: 3.3 years. Calculation: Target leverage: 3.0x × $100M EBITDA = $300M debt. Debt to pay down: $500M - $300M = $200M. Annual debt paydown: $60M FCF. Years required: $200M / $60M = 3.33 years. Note: This assumes EBITDA stays flat at $100M (conservative). If EBITDA grows, deleveraging happens faster because: (1) More FCF generated (higher EBITDA → higher FCF), (2) Denominator increases (3.0x leverage on $120M EBITDA = $360M allowed debt, less paydown needed). Reality: Most LBOs have both debt paydown AND EBITDA growth, so deleveraging from 5.0x to 3.0x typically takes 2-3 years (faster than flat-EBITDA calculation). Option 2 (5 years) would be if annual paydown was only $40M. Option 3 (2 years) would require $100M annual paydown (need higher FCF). Option 4 (8.3 years) is nonsensical given the numbers.',
  },
  {
    id: 'lbo-mc-4',
    question: 'Which of the following industries is BEST suited for a leveraged buyout?',
    options: ['Early-stage biotech (pre-revenue, high R&D burn)', 'Mature manufacturing with stable cash flows and asset base', 'High-growth SaaS requiring significant reinvestment', 'Commodity producer with volatile revenues'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Mature manufacturing with stable cash flows. Ideal LBO characteristics: (1) **Stable, predictable cash flows**: Can reliably service debt (interest + amortization). Manufacturing, distribution, business services fit this profile. (2) **Asset base for collateral**: Tangible assets (PP&E, inventory, receivables) secure debt. Lenders comfortable with asset-backed loans. (3) **Low CapEx requirements**: Mature businesses need maintenance CapEx only (3-5% of revenue), not growth CapEx. More FCF available for debt paydown. (4) **Defensible market position**: Limited disruption risk (not tech-driven, not commodity-exposed). (5) **Opportunity for operational improvements**: PE can professionalize management, implement best practices, drive margin expansion. Option 1 (biotech): Terrible for LBO. No cash flows (burning cash), no collateral, binary outcomes (drug succeeds or fails). Requires equity financing, not debt. Option 3 (high-growth SaaS): Problematic. Need to reinvest FCF in growth (sales, R&D, infrastructure), not debt paydown. Can work at lower leverage (2-3x), but not 5-6x. Option 4 (commodity): Too volatile. EBITDA swings with oil/steel/grain prices. In downturn, can\'t service debt (defaults). Lenders avoid cyclical commodity exposure.',
  },
  {
    id: 'lbo-mc-5',
    question: 'A PE firm structures an LBO with $300M equity and $600M debt. At exit, equity value is $900M. The firm attributes $400M of gains to debt paydown, $350M to EBITDA growth, and $150M to multiple expansion. Which source should be viewed most skeptically?',
    options: ['Debt paydown (most of value creation)', 'EBITDA growth (requires execution risk)', 'Multiple expansion (market dependent)', 'All three are equally reliable'],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: Multiple expansion is least reliable. Analysis: Total gains: $900M - $300M = $600M. Breakdown: Debt paydown $400M (67%), EBITDA growth $350M (58%), Multiple expansion $150M (25%). [Note: these overlap—debt paydown converts EV to equity; EBITDA growth increases EV]. Reliability ranking: (1) **Debt paydown** (most reliable): Mechanical process. Generate FCF → pay down debt → equity value increases. Fully in company\'s control. Risk: only if can\'t generate projected FCF. (2) **EBITDA growth** (moderately reliable): Requires execution (cost cuts, revenue growth, margin expansion). PE firms have playbook for this (performance improvement, add-on acquisitions). Risk: execution failure, competition, market deterioration. (3) **Multiple expansion** (least reliable): Depends on market conditions (rates, sentiment, sector trends). Company has ZERO control. 2021: 12x EBITDA multiples (frothy), 2022: 7x multiples (compressed). Risk: Multiple compression can erase all operational gains. Underwriting principle: NEVER assume multiple expansion in base case. Use exit multiple = entry multiple (or lower). If entry at 7x, assume exit at 7x. If multiples expand to 8x, treat as upside/luck. The $150M from multiple expansion in this deal should be discounted—was it luck (market went up) or skill? If IRR targets require multiple expansion, walk away (poor deal). Build deals on debt paydown + EBITDA growth only.',
  },
];
