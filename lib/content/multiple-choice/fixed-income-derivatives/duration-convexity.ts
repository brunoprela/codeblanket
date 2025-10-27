import { MultipleChoiceQuestion } from '@/lib/types';

export const durationConvexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'dc-mc-1',
    question:
      'A bond has Macaulay duration of 8 years and currently yields 5%. What is its modified duration?',
    options: ['7.62', '8.00', '8.40', '7.50'],
    correctAnswer: 0,
    explanation:
      "Modified Duration = Macaulay Duration / (1 + y). Calculation: Mod_Duration = 8 / (1 + 0.05) = 8 / 1.05 = 7.619 ≈ 7.62 years. Why divide by (1+y)? Modified duration converts Macaulay duration (weighted average time) into price sensitivity measure. The adjustment accounts for compounding. Interpretation: If yields rise 1% (from 5% to 6%), bond price will fall approximately 7.62%. More precisely: ΔP/P ≈ -7.62 × 0.01 = -7.62% (negative because inverse relationship). Why not 8.00? That's Macaulay duration, not modified. Macaulay measures timing of cash flows, modified measures price sensitivity. Example: $1,000 bond with mod duration 7.62, yield rises 0.50% (50bp). Expected price drop: ΔP = -7.62 × 0.005 × $1,000 = -$38.10. New price ≈ $961.90. Important: Modified duration assumes parallel yield curve shift and small yield changes. For large changes, convexity adjustment needed. Real-world application: Portfolio managers use modified duration for interest rate risk management. Target duration = 0 means immunized (hedged). Duration = 5 means moderate risk. Duration = 10 means high risk (suitable for rates expected to fall).",
  },
  {
    id: 'dc-mc-2',
    question:
      'A $1,000 bond has modified duration of 6.5. What is its approximate DV01?',
    options: ['$0.65', '$6.50', '$65.00', '$0.065'],
    correctAnswer: 0,
    explanation:
      'DV01 = Modified Duration × Price × 0.0001. Calculation: DV01 = 6.5 × $1,000 × 0.0001 = 6.5 × 0.1 = $0.65. Meaning: Bond price changes by $0.65 for every 1 basis point (0.01%) change in yield. Verification: If yield changes +1bp (0.01%), price change = -6.5 × 0.0001 × $1,000 = -$0.65. If yield changes +100bp (1%), price change = -6.5 × 0.01 × $1,000 = -$65. Why 0.0001 multiplier? DV01 = "Dollar Value of 01" where 01 = 1 basis point = 0.01% = 0.0001 in decimal. Why not $6.50? That would be for 10bp change, not 1bp. Formula: DV01 = D × P × 0.0001, not D × P × 0.001. Why not $65? That\'s the change for 100bp (1%), not 1bp. Real-world usage: Portfolio manager has $100M portfolio with aggregate DV01 = $100,000. If Fed raises rates 25bp, expected loss = $100,000 × 25 = $2.5M. To hedge: Need to short bonds/futures with equivalent DV01. If 10-year Treasury future has DV01 = $85 per contract, hedge = $100,000 / $85 = 1,176 contracts. Trading: DV01 used for position sizing. If max loss tolerance = $10,000 for 25bp move, max DV01 = $10,000 / 25 = $400. Can then calculate max bond position. Important: DV01 assumes linear relationship (duration only). For large yield changes, add convexity adjustment.',
  },
  {
    id: 'dc-mc-3',
    question:
      'Why is convexity considered a desirable property for a bond investor?',
    options: [
      'Positive convexity means gains are larger than losses for equal yield changes',
      'Convexity reduces credit risk',
      'Higher convexity means higher yield',
      'Convexity eliminates interest rate risk',
    ],
    correctAnswer: 0,
    explanation:
      'Positive convexity creates asymmetric payoff: gains > losses. Example with 10-year bond, modified duration = 7, convexity = 50: Scenario 1: Yields fall 1% (from 6% to 5%): Duration estimate: +7% price gain, Convexity adjustment: +0.5 × 50 × (0.01)² = +0.25%, Total gain: 7.25% (convexity adds to gain). Scenario 2: Yields rise 1% (from 6% to 7%): Duration estimate: -7% price loss, Convexity adjustment: +0.5 × 50 × (0.01)² = +0.25%, Total loss: -6.75% (convexity cushions loss). Result: Same 1% yield change → gain 7.25% vs lose 6.75% (asymmetric!). Why beneficial? Upside > downside for equal moves. Like owning a free option. In volatile markets, positive convexity adds value (benefit from rate swings either direction). Mathematical reason: Price-yield relationship is CURVED (convex), not linear. Second derivative (convexity) captures this curvature. Why not "reduces credit risk"? Convexity is about interest rate risk, not credit risk. Credit risk = default probability, unrelated to convexity. Why not "higher yield"? Actually inverse: Higher convexity bonds typically have LOWER yields (investors pay premium for convexity). 30-year Treasuries have high convexity but lower yield than some corporate bonds. Why not "eliminates risk"? Convexity reduces interest rate risk but doesn\'t eliminate it. Duration + convexity together approximate price changes, but not perfect. For very large yield changes (>200bp), even second-order approximation breaks down. Real-world trading: Investors seek positive convexity strategies. Barbell strategy (short + long bonds) has MORE convexity than bullet (medium bonds) with same duration. During 2008 crisis, long Treasuries gained 25%+ (both duration and convexity benefits). Portfolio managers: Prefer positive convexity but pay for it (lower yield). Trade-off: Yield vs convexity protection.',
  },
  {
    id: 'dc-mc-4',
    question:
      'A portfolio has bonds worth $5M with duration 5 and $3M with duration 10. What is the portfolio duration?',
    options: ['6.875', '7.50', '15.00', '5.00'],
    correctAnswer: 0,
    explanation:
      "Portfolio duration = weighted average by market value. Formula: D_portfolio = Σ(w_i × D_i) where w_i = MV_i / Total_MV. Calculation: Total MV = $5M + $3M = $8M. Weight bond 1: w_1 = $5M / $8M = 0.625 (62.5%). Weight bond 2: w_2 = $3M / $8M = 0.375 (37.5%). Portfolio duration = 0.625 × 5 + 0.375 × 10 = 3.125 + 3.750 = 6.875 years. Interpretation: Portfolio behaves like a single bond with 6.875-year duration. If yields rise 1%, portfolio price falls approximately 6.875%. Verification: Bond 1 loss: $5M × 5 × 0.01 = $250,000. Bond 2 loss: $3M × 10 × 0.01 = $300,000. Total loss: $550,000 on $8M = 6.875% ✓. Why not 7.50? That would be simple average (5+10)/2, ignoring different position sizes. Wrong because bond 2 is smaller position. Why not 15? That's the sum of durations, not weighted average. Nonsensical (portfolio can't have duration > any component). Why not 5? That's just bond 1's duration, ignoring bond 2 entirely. Real-world application: Fund managers target specific portfolio duration based on interest rate view. Bullish (expect rates to fall): Increase duration to 8-10 (amplify gains). Bearish (expect rates to rise): Decrease duration to 2-3 (reduce losses). Neutral: Match benchmark duration ±0.5. Rebalancing: To increase portfolio duration from 6.875 to 8: Buy more of bond 2 (duration 10) or add new long-duration bonds. To decrease to 5: Sell bond 2, buy shorter-duration bonds or cash. Calculation tool: Portfolio duration × Portfolio Value × 0.0001 = Portfolio DV01. Example: 6.875 × $8M × 0.0001 = $5,500 DV01. Means $5,500 loss for each 1bp yield rise. Important: This assumes parallel yield curve shift. Non-parallel shifts (steepening/flattening) require key rate duration analysis.",
  },
  {
    id: 'dc-mc-5',
    question:
      'A pension fund has liabilities with duration 12 years. To immunize, the fund should construct an asset portfolio with duration of:',
    options: [
      '12 years (match liability duration)',
      '0 years (eliminate interest rate risk)',
      '20 years (exceed liability duration for safety)',
      '6 years (half the liability duration)',
    ],
    correctAnswer: 0,
    explanation:
      "Immunization requires matching asset duration to liability duration. Principle: If D_assets = D_liabilities, interest rate changes affect both equally (net effect = zero). Example: Liabilities PV = $100M, duration = 12 years. If yields rise 1%: Liability PV falls: ΔL = -12 × 0.01 × $100M = -$12M, new PV = $88M. If asset duration = 12: Asset PV falls: ΔA = -12 × 0.01 × $100M = -$12M. Net funded status change = $0 (immunized!). If asset duration ≠ 12: Mismatch creates risk. Example: Asset duration = 8 years. Yields rise 1%: Assets fall $8M, liabilities fall $12M, funded status improves $4M (lucky but risky). Yields fall 1%: Assets rise $8M, liabilities rise $12M, funded status worsens $4M (underfunded). Why not 0 years? Duration = 0 means assets don't respond to rate changes (e.g., cash). But liabilities still have duration 12, so they change. Mismatch = 12 years of unhedged risk. Worst possible immunization. Why not 20 years? Exceeding liability duration creates OVER-hedging. If yields rise: Assets fall 20% × $100M = $20M, liabilities fall 12% × $100M = $12M, funded status worsens $8M (underfunded). Not immunized - actually amplifies risk. Why not 6 years? Half-hedged = still exposed. Gap = 12 - 6 = 6 years of duration risk. Real-world implementation: Pension funds: Calculate liability duration from projected benefit payments (typically 10-20 years). Buy bond portfolio matching that duration (mix of Treasuries). Rebalance quarterly (duration drifts as time passes and yields change). Insurance companies: Similar strategy for policy liabilities. Liability duration = 15 years → asset portfolio duration = 15 years. Dynamic adjustment: As liabilities age, duration shortens naturally. Must rebalance assets accordingly. If liability duration drops to 10 years, sell long bonds, buy medium bonds. Limitations of simple duration matching: Only works for PARALLEL yield curve shifts (all rates move equally). Non-parallel shifts (steepening/flattening) break immunization. Solution: Match key rate durations at multiple maturities (not just overall duration). Better approach: Convexity matching too (not just duration). Match both first and second derivatives.",
  },
];
