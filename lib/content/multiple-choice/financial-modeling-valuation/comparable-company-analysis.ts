import { MultipleChoiceQuestion } from '@/lib/types';

export const comparableCompanyAnalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cca-mc-1',
    question: 'When calculating Enterprise Value for a comps analysis, which of the following items should be ADDED to market capitalization?',
    options: [
      'Cash and cash equivalents',
      'Total debt (short-term + long-term)',
      'Accounts receivable',
      'Retained earnings',
    ],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Total debt should be ADDED to market cap to get Enterprise Value. Formula: EV = Market Cap + Debt - Cash + Minority Interest + Preferred Stock. Logic: Market cap represents value of equity. To get total firm value (EV), add debt claims (debt holders also have claim on company). Then subtract cash (belongs to equity, effectively reduces net debt). Option 1 (cash) is subtracted, not added. Option 3 (AR) is already in market cap (asset on balance sheet). Option 4 (retained earnings) is equity account, already in market cap. Common mistake: Adding cash instead of subtracting. Cash is asset company owns, reduces net obligations. Example: Company worth $10B equity, has $2B debt and $500M cash. EV = $10B + $2B - $500M = $11.5B. This represents operating business value before financing.',
  },
  {
    id: 'cca-mc-2',
    question: 'You are valuing a company with $500M EBITDA using comps. The comparable companies have EV/EBITDA multiples of: 18x, 20x, 22x, 24x, 45x. What is the most appropriate valuation approach?',
    options: [
      'Use the mean (25.8x) to value the company at $12.9B EV',
      'Use the median (22x) to value the company at $11.0B EV',
      'Exclude the 45x outlier, then use mean of remaining (21x) for $10.5B EV',
      'Use the highest multiple (45x) to maximize valuation at $22.5B EV',
    ],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: Exclude the obvious outlier (45x is 2x the next highest), then use mean or median of remaining comps. The 45x comp is likely: (1) Different business model, (2) Acquisition target with takeover premium, (3) Bubble stock, (4) Wrongly included—should be removed. After removing outlier: remaining multiples are 18x, 20x, 22x, 24x. Mean = 21x, Median = 21x (same). EV = $500M × 21x = $10.5B. Option 1 (mean of 25.8x) is distorted by outlier—pulls average up 23% from one data point. Option 2 (median 22x) is more robust than mean but still affected by outlier presence (median is middle of 5, which is 22x). Option 4 (45x) is cherry-picking highest to maximize value—intellectually dishonest and would be rejected by any buyer/audit. Best practice: (1) Investigate outlier—why so high? (2) If no justification, exclude. (3) Use median of clean data (robust) or mean if symmetric distribution. (4) Show range (18x-24x) not just point estimate.',
  },
  {
    id: 'cca-mc-3',
    question: 'Two SaaS companies both trade at 10x EV/Revenue. Company A grows 40% annually with 15% EBITDA margin. Company B grows 20% with 30% EBITDA margin. Which statement is most accurate?',
    options: [
      'The companies are fairly valued relative to each other at the same multiple',
      'Company A deserves a higher multiple due to superior growth; it is undervalued',
      'Company B deserves a higher multiple due to superior profitability; it is undervalued',
      'Both are likely overvalued since 10x revenue is high for any SaaS company',
    ],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Company A is undervalued relative to Company B. In SaaS valuations, GROWTH typically commands higher premium than MARGINS, especially at high growth rates. Rationale: Company A (40% growth, 15% margin): "Land grab" phase, investing heavily for growth (suppressing margins). Market rewards hypergrowth companies—if you double in 2 years, margin improvement comes later. Company B (20% growth, 30% margin): More mature, optimized for profitability. Slower growth means lower future upside. Multiple decomposition: EV/Revenue = EV/EBITDA × EBITDA/Revenue. Company A: 10x = EV/EBITDA × 0.15 → EV/EBITDA = 67x. Company B: 10x = EV/EBITDA × 0.30 → EV/EBITDA = 33x. Company A trades at 67x EBITDA vs Company B at 33x—market is paying 2x for A\'s growth. If both trade at 10x revenue, market isn\'t sufficiently rewarding A\'s superior growth. Typical premium: 40% growth commands 12-15x revenue, 20% growth commands 7-10x revenue. If both at 10x, Company A should be 12x+ (undervalued) and Company B is fairly valued or overvalued. Option 1 is wrong—different fundamentals should trade at different multiples. Option 3 is wrong—Company B\'s higher margins don\'t override Company A\'s much higher growth in SaaS sector. Option 4 may be true in bear market but doesn\'t address relative valuation.',
  },
  {
    id: 'cca-mc-4',
    question: 'When using Last Twelve Months (LTM) financials for comps analysis, which of the following is the PRIMARY advantage over using forward estimates?',
    options: [
      'LTM data is actual reported results, not subjective forecasts',
      'LTM data always shows higher growth rates than forward estimates',
      'LTM multiples are always lower than forward multiples',
      'LTM data includes one-time items that improve comparability',
    ],
    correctAnswer: 0,
    explanation: 'Option 1 is correct: LTM uses actual reported financials (objective, audited) rather than analyst estimates (subjective, may be biased). Advantage: No forecast risk. LTM is verifiable fact from 10-Q/10-K filings. Forward estimates can be wrong—analysts may be too optimistic (inflate revenue projections) or pessimistic. Particularly important for M&A fairness opinions where precision and defensibility matter. Disadvantages of LTM: (1) Backward-looking—doesn\'t capture improving/deteriorating trends. (2) May include one-time items (restructuring, asset sales) that distort. (3) For rapidly growing companies, understates current trajectory. Option 2 is false—LTM doesn\'t systematically show higher growth (depends on business cycle). Option 3 is false—LTM multiples are typically HIGHER than forward (denominator is lower—past revenue < future revenue, so multiple is higher). Forward P/E often used because it\'s lower (better optics). Option 4 is false—one-time items reduce comparability (should be adjusted out in both LTM and forward). Best practice: Show both. LTM for reliability, Forward (NTM = Next Twelve Months) for growth trajectory. If LTM P/E = 30x and Forward P/E = 24x, implies 25% earnings growth expected.',
  },
  {
    id: 'cca-mc-5',
    question: 'You include a competitor in your comps group, but discover it was acquired 2 weeks ago at a 35% premium to its prior trading price. How should you handle this in your trading comps analysis?',
    options: [
      'Use the pre-acquisition trading price to avoid distortion from takeover premium',
      'Use the acquisition price since it represents current market value',
      'Exclude the company entirely from the trading comps analysis',
      'Average the pre- and post-acquisition prices for a blended valuation',
    ],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: Exclude the acquired company from TRADING comps (but include in PRECEDENT TRANSACTION comps). Reasoning: Trading comps should reflect standalone public market trading multiples. Post-acquisition price includes: (1) Takeover premium (typically 25-40%), (2) Synergies buyer expects to realize, (3) Strategic value (may overpay), (4) Control premium (acquiring 100% vs buying minority shares). These are M&A phenomena, not trading multiples. Including inflates your comps by 30%+. Option 1 (pre-acquisition price) is tempting but problematic—price was depressed anticipating acquisition, or acquisition leaked. Not representative of true trading level. Option 2 (acquisition price) belongs in precedent transactions analysis (Section 5), NOT trading comps. Option 4 (average) has no theoretical justification—mixing apples and oranges. Correct approach: (1) Remove from trading comps table. (2) Add to precedent transactions table (separate analysis). (3) In presentation note: "Comp X was acquired on [date] at [price], [%] premium. Excluded from trading comps, included in transaction comps." (4) If it was your best comp and now gone, find replacement or reduce comp set size. Market structure note: As companies get acquired, comp groups shrink. M&A waves can make comps analysis harder (fewer standalone traders). This happened in biotech (2018-2020) and SPACs (2020-2021)—best comps kept getting taken out.',
  },
];
