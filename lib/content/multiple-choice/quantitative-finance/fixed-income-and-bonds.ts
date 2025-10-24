import type { MultipleChoiceQuestion } from '@/lib/types';

export const fixedIncomeAndBondsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fib-mc-1',
    question:
      'A 10-year bond with 6% annual coupon is trading at $1,050 (face value $1,000). If yields rise 0.5%, the bond price falls to $1,010. What is the approximate modified duration of this bond?',
    options: ['4.0 years', '7.6 years', '8.0 years', '10.0 years'],
    correctAnswer: 1,
    explanation:
      "Modified duration formula: D_mod ≈ -ΔP/P / Δy. Price change: ΔP = $1,010 - $1,050 = -$40. Percentage change: ΔP/P = -$40 / $1,050 = -0.0381 = -3.81%. Yield change: Δy = +0.005 (0.5% = 50 bps). Modified duration: D_mod = -(-0.0381) / 0.005 = 0.0381 / 0.005 = 7.62 ≈ 7.6 years. INTERPRETATION: For every 1% yield increase, bond loses approximately 7.6% of value. 10-year coupon bond has duration < 10 years because coupons paid before maturity reduce weighted average time to cash flows. Zero-coupon 10Y bond would have duration = 10 years. Option A (4.0) too low - that would be ~5-year bond. Option C (8.0) close but calculation gives 7.6. Option D (10.0) wrong - that's maturity, not duration (only true for zero-coupon).",
  },
  {
    id: 'fib-mc-2',
    question:
      'An investor holds a bond portfolio with modified duration of 6.0 and convexity of 80. If yields fall by 2% (200 bps), what is the approximate total percentage price change?',
    options: [
      '+12.0% (duration effect only)',
      '+13.6% (duration + convexity)',
      '+10.4% (duration effect reduced by convexity)',
      '+18.0% (duration × 2 + convexity)',
    ],
    correctAnswer: 1,
    explanation:
      'Price change formula with convexity: ΔP/P ≈ -D_mod × Δy + 0.5 × C × (Δy)². Given: D_mod = 6.0, C = 80, Δy = -0.02 (yields FALL 2%). Duration effect: -6.0 × (-0.02) = +0.12 = +12.0%. Convexity effect: 0.5 × 80 × (-0.02)² = 0.5 × 80 × 0.0004 = 0.016 = +1.6%. Total: 12.0% + 1.6% = +13.6%. KEY INSIGHT: Convexity is ALWAYS beneficial (positive). When yields fall, convexity ADDS to gains (12% → 13.6%). When yields rise, convexity REDUCES losses. This asymmetry makes positive convexity valuable (pay premium for high-convexity bonds). Option A ignores convexity (linear approximation only). Option C has sign wrong - convexity adds, not reduces. Option D incorrectly calculates convexity contribution.',
  },
  {
    id: 'fib-mc-3',
    question:
      'The yield curve currently shows: 2Y = 4.0%, 10Y = 4.8%, 10Y-2Y spread = 80 bps. Which statement about yield curve strategies is MOST accurate?',
    options: [
      'A curve steepener trade (long 10Y, short 2Y) has positive carry because 10Y yields more than 2Y',
      'A curve flattener trade (short 10Y, long 2Y) benefits if the Fed cuts short-term rates',
      'A duration-neutral curve steepener has negative carry because the short position notional exceeds the long position',
      'Roll-down return is highest for 2Y bonds because they mature sooner',
    ],
    correctAnswer: 2,
    explanation:
      'Duration-neutral steepener mechanics: Long 10Y (D ≈ 8.5), Short 2Y (D ≈ 1.9). To make duration-neutral: Long $100M × 8.5 = Short $X × 1.9 → X = $447M (short 4.47× the long!). Carry calculation: Earn: $100M × 4.8% = $4.8M from 10Y. Pay: $447M × 4.0% = $17.9M from 2Y short. NET CARRY: $4.8M - $17.9M = -$13.1M (NEGATIVE!). Why? Must short MUCH MORE 2Y (lower duration) to offset 10Y (higher duration) → pay more in short interest than earn in long coupons. This is typical for duration-neutral curve trades - positive carry only if long the SHORT end, negative carry if long the LONG end. Option A wrong: Simple steepener (not duration-neutral) does have positive carry, but question implies duration-neutral (standard for curve trades). Option B wrong: Flattener benefits if long rates fall MORE than short rates, not just "Fed cuts" (Fed cuts help steepener, not flattener). Option D wrong: Roll-down highest when curve is steep AND duration is long (10Y rolls down more than 2Y on steep curve).',
  },
  {
    id: 'fib-mc-4',
    question:
      'A BBB-rated corporate bond trades at 150 bps spread over Treasuries. The company is downgraded to BB (high yield), and the spread widens to 400 bps. If the bond has modified duration of 5.0, what is the approximate price impact from the downgrade?',
    options: [
      '-12.5% (spread widening of 250 bps × duration 5.0)',
      '-2.5% (spread change 400 - 150 = 250 bps)',
      '-15.0% (new spread 400 bps × duration 5.0 / 100)',
      '-20.0% (typical downgrade impact for fallen angels)',
    ],
    correctAnswer: 0,
    explanation:
      'Credit spread impact calculation: Spread widening: 400 bps - 150 bps = 250 bps = 0.025 (2.5%). Price change: ΔP/P ≈ -D_mod × Δy = -5.0 × 0.025 = -0.125 = -12.5%. INTERPRETATION: When credit spreads widen by 250 bps, and duration is 5.0, bond loses 12.5% of value. This is the "spread duration" effect - same calculation as interest rate duration, applied to spread changes. Reality check: BBB → BB downgrades ("fallen angels") typically cause 10-15% price drops (matches -12.5%). Components: (1) Spread widens 250 bps → -12.5% from spread duration, (2) Forced selling by institutions (restricted to investment grade) → additional liquidity pressure. Total impact often exceeds simple duration calculation due to technical factors. Option B wrong: Can\'t just use spread change in bps - must multiply by duration. Option C wrong: Incorrect formula (uses new spread level, not change). Option D wrong: -20% is extreme (more typical for BB → CCC downgrade or near-default).',
  },
  {
    id: 'fib-mc-5',
    question:
      'An investor wants to immunize a $100M liability due in 10 years by buying bonds. Which strategy provides the BEST immunization against interest rate risk?',
    options: [
      'Buy $100M face value of 10-year coupon bonds (duration 7.5 years)',
      'Buy $100M face value of 10-year zero-coupon bonds (duration 10 years)',
      'Buy a mix of 5-year and 15-year bonds with weighted average duration of 10 years',
      'Buy 10-year Treasury Inflation-Protected Securities (TIPS) with duration 10 years',
    ],
    correctAnswer: 1,
    explanation:
      'Immunization requires EXACT duration matching. Liability: $100M due in 10 years → duration = 10 years (zero-coupon-like). OPTION A (10Y coupon bonds, D=7.5): Duration mismatch! Bond duration (7.5) < liability duration (10). If rates rise: Bond loses 7.5% per 1% rate rise. Liability discount increases → present value falls MORE (10% per 1% rate rise). Net: Shortfall (underhedged). If rates fall: Bond gains 7.5%, liability PV rises 10% → still shortfall. Conclusion: Fails to immunize. OPTION B (10Y zero-coupon, D=10): PERFECT MATCH. Bond duration = liability duration = 10 years. Zero-coupon structure matches liability structure (single payment at maturity). If rates rise/fall: Bond and liability move in LOCKSTEP → hedge effective. Reinvestment risk: ZERO (no coupons to reinvest). This is textbook immunization. OPTION C (5Y + 15Y barbell, avg D=10): Duration matches, BUT: Convexity higher than Option B (barbell > bullet). Reinvestment risk: 5Y bond matures early, must reinvest for 5 years (rate uncertainty). Rebalancing: As bonds age, duration drifts → need periodic adjustment. Conclusion: Works but INFERIOR to Option B (adds unnecessary complexity and risk). OPTION D (10Y TIPS, D=10): Duration matches, but TIPS protect INFLATION risk, not interest rate risk. Immunization typically refers to NOMINAL interest rate risk hedging. If liability is inflation-indexed, TIPS would be appropriate. But for standard nominal liability, TIPS add inflation exposure mismatch. Conclusion: Zero-coupon Treasury (Option B) is BEST - exact duration match, zero reinvestment risk, simplest implementation.',
  },
];
