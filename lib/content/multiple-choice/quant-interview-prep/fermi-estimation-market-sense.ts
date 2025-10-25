import { MultipleChoiceQuestion } from '@/lib/types';

export const fermiEstimationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fe-mc-1',
    question:
      "You need to estimate Netflix's subscriber count but don't know the exact number. Which approach gives the MOST accurate estimate?",
    options: [
      'Multiply US population by estimated penetration rate and extrapolate globally',
      'Use revenue ($30B) divided by average revenue per user ($15/month × 12) to get subscribers',
      'Estimate by region (US, Europe, LatAm, Asia) with different penetration rates, then sum',
      'Compare to other streaming services and estimate market share',
    ],
    correctAnswer: 2,
    explanation:
      'Regional approach (option 3) is most accurate because penetration rates vary dramatically: US/Canada ~45%, Europe ~30%, LatAm ~20%, Asia ~8%. This accounts for market maturity, income levels, and competition. Simple US extrapolation (option 1) fails because US is saturated. Revenue method (option 2) assumes constant ARPU globally (false: US pays $15/mo, India pays $3/mo). Market share approach (option 4) requires knowing total streaming market, which is also unknown. Regional breakdown with different assumptions gives: US/Canada 65M + Europe 75M + LatAm 40M + Asia 45M = 225M (close to actual 247M).',
  },
  {
    id: 'fe-mc-2',
    question:
      'When estimating transaction costs for a $100M equity order, which factor typically contributes MOST to total cost?',
    options: [
      'Bid-ask spread (typically 5-10 basis points)',
      'Exchange and clearing fees (typically 0.5-1 bp)',
      'Market impact from large order size (typically 50-100 bps for 40% of daily volume)',
      'Broker commission (typically 2-3 bps)',
    ],
    correctAnswer: 2,
    explanation:
      'Market impact dominates for large orders. At 40% of daily volume, market impact can be 50-100 bps ($500K-$1M on $100M order) due to price movement while executing. Bid-ask spread (~5-10 bps = $50-100K) is paid once. Exchange fees (~0.5 bp = $5K) are tiny. Broker commission (~2-3 bps = $20-30K) is fixed. The square-root market impact model: impact ∝ σ × sqrt(order_size/daily_volume) means large orders face quadratically increasing costs. For comparison, a $1M order (0.04% of daily volume) might only pay 5-10 bps total, but a $100M order (40× larger) pays ~20× more per dollar in market impact due to nonlinear scaling.',
  },
  {
    id: 'fe-mc-3',
    question:
      'You estimate the number of active retail traders in the US as 12 million with average account size $50K. How should you sanity-check this estimate?',
    options: [
      'Compare total retail capital ($600B) to total US equity market cap ($50T) - should be ~1-2%',
      'Check if 12M is reasonable relative to US population and brokerage account statistics',
      'Verify against known data: major brokers (Schwab, Fidelity, Robinhood) report ~30M total accounts, with ~40% active',
      'All of the above - use multiple independent checks',
    ],
    correctAnswer: 3,
    explanation:
      'Always use multiple sanity checks! (1) Capital check: $600B retail vs $50T total market cap = 1.2% - reasonable since institutions dominate. (2) Population check: 12M active traders out of 330M total (3.6%) or 200M working adults (6%) - plausible given many don\'t invest. (3) Broker data: Schwab (33M accounts), Fidelity (40M), Robinhood (23M), others (30M+) = ~80-100M total brokerage accounts. If 40% are "active" (trade quarterly), that\'s 30-40M. Our 12M for very active (weekly traders) is ~30-40% of all active, which fits. Cross-validation with multiple methods catches errors and increases confidence. If all checks pass, estimate is probably within 2× of true value.',
  },
  {
    id: 'fe-mc-4',
    question:
      'When using the Kelly Criterion to size positions (f* = (pb - q)/b), which statement is TRUE?',
    options: [
      'You should always bet exactly the Kelly fraction to maximize long-term growth',
      'Half-Kelly (50% of Kelly) is often preferred in practice despite lower expected growth',
      'Kelly only works for positive expected value bets (pb > q)',
      'Both B and C are correct',
    ],
    correctAnswer: 3,
    explanation:
      "Both B and C are correct. (C) Kelly requires positive edge: if pb < q (negative EV), Kelly gives negative f*, meaning don't bet. For zero EV (pb = q), f* = 0 (don't bet). (B) Half-Kelly is preferred in practice because: (1) reduces volatility by 50% while maintaining 75% of full Kelly growth, (2) more robust to estimation errors in p and b (full Kelly assumes perfect knowledge), (3) prevents ruin from brief unlucky streaks. Example: p=0.55, b=1.25 gives full Kelly f*=19% of capital. But if true p=0.52 (estimation error), full Kelly leads to excessive risk. Half-Kelly at 9.5% is safer. (A) is false: full Kelly maximizes geometric mean but has high volatility and risk of large drawdowns. Professional traders typically use 25-50% of Kelly (fractional Kelly).",
  },
  {
    id: 'fe-mc-5',
    question:
      'You estimate that high-frequency trading accounts for 50% of US equity volume. Which follow-up question is MOST important for understanding HFT profitability?',
    options: [
      'What is the average profit per share traded by HFT firms?',
      'What percentage of HFT volume is market making vs directional trading?',
      'How has HFT volume share changed over the past 5 years?',
      'What is the geographic distribution of HFT activity?',
    ],
    correctAnswer: 0,
    explanation:
      "Average profit per share (option 1) is the KEY metric for profitability calculation. With 50% of $500B daily volume = $250B, if HFT makes $0.0001 per share (very small edge), that's still $25M daily profit. But if they make $0.00005/share, profit halves to $12.5M. This multiplier effect makes profit per share the critical variable. Strategy mix (option 2) matters for risk profile but doesn't directly give profitability. Historical trends (option 3) are interesting but don't answer current profitability. Geography (option 4) affects infrastructure costs but is secondary. The profitability formula is: Total Profit = Volume × Profit/Share. We know volume (~$250B), so profit/share determines everything. Market makers might earn $0.00005-0.0001/share (0.5-1 bp), while stat arb might earn $0.0002-0.0003/share but on lower volume.",
  },
];
