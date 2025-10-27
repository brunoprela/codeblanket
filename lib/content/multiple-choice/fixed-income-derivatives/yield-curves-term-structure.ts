import { MultipleChoiceQuestion } from '@/lib/types';

export const yieldCurvesTermStructureMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'ycts-mc-1',
      question:
        'You bootstrap a yield curve and find: 1-year spot = 3.0%, 2-year spot = 4.0%. What is the 1-year forward rate starting in 1 year (1y1y forward)?',
      options: ['5.01%', '3.50%', '4.00%', '7.00%'],
      correctAnswer: 0,
      explanation:
        'Forward rate formula: (1 + s₂)² = (1 + s₁) × (1 + f₁,₂). Solving for f₁,₂: f₁,₂ = [(1 + s₂)² / (1 + s₁)] - 1. Calculation: f₁,₂ = [(1.04)² / 1.03] - 1 = [1.0816 / 1.03] - 1 = 1.0501 - 1 = 0.0501 = 5.01%. Interpretation: If you invest for 2 years at 4%, you earn the same as investing for 1 year at 3% and then reinvesting for another year at 5.01%. This is the "break-even" rate. Why not 3.50% (simple average)? Forward rates are geometric, not arithmetic averages. The 2-year rate (4%) is higher than 1-year (3%), so the second year must have an even higher rate (5.01%) to average to 4%. Why not 7%? That would be (1.04)² - 1 = 8.16% total return, not the forward rate. Real-world application: Forward rates are used for FRAs (Forward Rate Agreements) and pricing interest rate swaps. If you need to lock in a 1-year borrowing rate starting in 1 year, the fair rate is 5.01%. Banks use this for lending decisions. Important: Forward rates reflect market expectations (under Expectations Hypothesis) or include liquidity premiums (under Liquidity Preference Theory). A rising forward curve (5.01% > 4% > 3%) suggests the market expects rates to increase, possibly due to inflation concerns or Fed tightening.',
    },
    {
      id: 'ycts-mc-2',
      question:
        'The 2yr-10yr Treasury spread is currently -15 basis points. What does this indicate?',
      options: [
        'Inverted yield curve - recession likely within 6-24 months',
        'Normal yield curve - healthy economy',
        'The 10-year yield is 15bp higher than 2-year (normal)',
        'Bond market malfunction',
      ],
      correctAnswer: 0,
      explanation:
        'Negative spread = inverted yield curve. Spread = 10yr yield - 2yr yield = -15bp means 2yr yield is 15bp HIGHER than 10yr yield. Example: If 10yr = 4.00% and 2yr = 4.15%, spread = 4.00% - 4.15% = -0.15% = -15bp. This is INVERTED. Historical significance: Every US recession since 1955 has been preceded by a 2yr-10yr inversion. Typical lead time: 6-24 months (median ~12 months). Recent examples: 2006 inversion → 2008-09 recession (22-month lead), 2019 inversion → 2020 recession (though COVID accelerated it), 2022-2023 inversion → ? (as of writing, monitoring for 2024 recession). Why inversions predict recessions: Short-term rates rise: Fed fighting inflation by raising policy rates, Long-term rates fall: Market expects Fed to cut rates in future (recession → easing), Spread compresses then inverts. Economic interpretation: Banks borrow short-term, lend long-term (profit on spread). When curve inverts, banks lose money on new loans → reduce lending → credit crunch → recession. Current -15bp inversion: Moderate depth (not extreme like -50bp in 2006), Still significant signal, Probability of recession in next 12-18 months: ~70-80% based on historical models. Trading implications: Defensive positioning: Reduce equity exposure, increase bonds, Sector rotation: Favor defensive sectors (utilities, consumer staples), Short duration bonds: If recession comes, Fed will cut rates (bond prices rise), Currency: USD typically strengthens during risk-off. Why not "Bond market malfunction"? Inversions are rational market behavior, not errors. They reflect collective wisdom of thousands of investors pricing in future rate cuts.',
    },
    {
      id: 'ycts-mc-3',
      question:
        'A 2-year Treasury note with 4% coupon trades at $100.50. Using only this bond, can you directly determine the 2-year spot rate?',
      options: [
        'No - you need the 1-year spot rate first to bootstrap',
        'Yes - YTM equals spot rate for all bonds',
        'Yes - spot rate = coupon rate when bond trades at premium',
        'No - you need at least 5 bonds to build a curve',
      ],
      correctAnswer: 0,
      explanation:
        "Cannot directly extract spot rate from a coupon bond alone. Reason: Coupon bonds have multiple cash flows at different times, each should be discounted at its own spot rate. 2-year bond with 4% coupon (semi-annual) has cash flows: $20 at 0.5yr, $20 at 1.0yr, $20 at 1.5yr, $1,020 at 2.0yr. Price = $20/(1+s₀.₅)^0.5 + $20/(1+s₁)^1 + $20/(1+s₁.₅)^1.5 + $1,020/(1+s₂)^2. To solve for s₂ (2-year spot), you need to know s₀.₅, s₁, and s₁.₅ first. Bootstrapping process: Start with T-Bill (zero-coupon) for shortest maturity → direct spot rate, Use that to help solve for next maturity's spot, Iteratively build up the curve. Why YTM ≠ Spot rate: YTM is the single discount rate that prices the bond, Spot rates are different rates for each cash flow, Only for zero-coupon bonds does YTM = spot rate (one cash flow). Example scenario: Suppose 1-year spot = 3%, 2-year spot = 4%, Bond YTM ≈ 3.95% (weighted average of spots), Spot rates more fundamental than YTM. Real-world implication: Bloomberg and other systems bootstrap curves from many bonds, Minimum inputs: 1 zero-coupon per major maturity OR several coupon bonds with different maturities, Industry standard: Use 10-15 Treasury securities spanning 3 months to 30 years. Common mistake: Assuming YTM = spot rate leads to arbitrage opportunities. Professional traders use spot rates (or forward rates) for pricing, not YTM. Important: This is why we need a complete yield curve, not just individual bond YTMs. Bootstrapping solves for the entire term structure of interest rates.",
    },
    {
      id: 'ycts-mc-4',
      question:
        'If the yield curve is "normal" (upward sloping), which theory BEST explains why it slopes upward?',
      options: [
        'Liquidity Preference Theory - investors demand higher returns for longer-term risk',
        'Expectations Hypothesis - investors expect rates to rise in the future',
        'Market Segmentation - different investors prefer different maturities',
        'Random Walk Theory - yields are unpredictable',
      ],
      correctAnswer: 0,
      explanation:
        'Liquidity Preference Theory best explains persistent upward slope. Key insight: Even if future rates are expected to stay flat, curve slopes up due to liquidity premium. Reasoning: Longer-term bonds are riskier (more interest rate risk = higher duration), Investors require compensation for tying up money longer, This premium increases with maturity → upward slope. Example: If 1-year rate = 3% and investors expect future 1-year rate to stay 3%, Pure Expectations says: 2-year rate should = 3% (average of 3% and 3%), But Liquidity Preference says: 2-year rate = 3% + liquidity premium (say 0.5%) = 3.5%, Result: Curve slopes upward even with flat rate expectations. Empirical evidence: Yield curve is upward-sloping ~70% of the time historically, Liquidity premiums are measurable (typically 0.3-0.5% per year of maturity), Even during periods when rates expected to fall slightly, curve often still slopes up (liquidity premium > negative expectations). Why not Expectations Hypothesis? EH assumes curve shape reflects future rate expectations only, If true, upward slope means rates expected to rise, But curve is upward-sloping even when survey data shows flat rate expectations, EH cannot explain persistent upward bias. Why not Market Segmentation? MS Theory says supply/demand in each maturity segment determines rates independently, While true that institutional preferences exist (banks like short-term, pensions like long-term), Doesn\'t explain why long-term consistently trades at premium, Moreover, arbitrageurs link segments (preventing wild mispricings). Historical context: Liquidity Preference introduced by Keynes (1930s), Later formalized by Hicks (1946), Modigliani-Sutch (1966) tested empirically → confirmed premiums exist. Real-world application: When pricing bonds, add term premium to expected rate path, For 10-year bond, might add 30-50bp "term premium" over expected average short rate, This is why "bonds earn more than cash on average" (term premium harvesting). Important: All three theories have merit, but Liquidity Preference explains the normal upward slope best. Expectations Hypothesis explains changes in slope (steepening vs flattening), Market Segmentation explains specific anomalies (like 7-10 year segment scarcity).',
    },
    {
      id: 'ycts-mc-5',
      question:
        'You need to price a 1.5-year bond, but only have spot rates for 1-year (3%) and 2-year (4%). Which interpolation method is most appropriate?',
      options: [
        'Linear interpolation on spot rates: s₁.₅ = 3.5%',
        'Linear interpolation on discount factors',
        'Linear interpolation on forward rates',
        'Cubic spline interpolation',
      ],
      correctAnswer: 1,
      explanation:
        "Linear interpolation on DISCOUNT FACTORS is most appropriate for no-arbitrage. Reason: Discount factors are the fundamental building blocks of valuation, They represent present value of $1 at each maturity, Linear interpolation on DFs preserves no-arbitrage conditions, While linear on rates can create arbitrage opportunities. Process: Calculate discount factors: DF₁ = 1/(1.03)^1 = 0.9709, DF₂ = 1/(1.04)^2 = 0.9246, Linear interpolate for 1.5yr: DF₁.₅ = DF₁ + 0.5 × (DF₂ - DF₁) = 0.9709 + 0.5 × (0.9246 - 0.9709) = 0.9709 - 0.0232 = 0.9477, Back out spot rate: (1 + s₁.₅)^1.5 = 1/DF₁.₅ = 1/0.9477 = 1.0552, s₁.₅ = 1.0552^(1/1.5) - 1 = 3.59% (not exactly 3.5%!). Why not linear on spot rates? Simple: s₁.₅ = 0.5 × (3% + 4%) = 3.5%, But this can create arbitrage: If you can replicate a 1.5yr cash flow using 1yr and 2yr cash flows, pricing must be consistent, Linear on rates doesn't guarantee this. Why linear on DFs works: No-arbitrage: Any cash flow can be priced as linear combination of DFs, Mathematical consistency: Linear interpolation on DFs = linear interpolation on log(1+rate) ≈ better rate behavior, Industry practice: Most systems interpolate DFs or forward rates, not spot rates. Why not cubic spline? Cubic spline is smoother and visually appealing, But for short gaps (0.5yr), linear sufficient, Cubic can introduce oscillations (Runge's phenomenon) with sparse data, More complexity than needed for 1.5yr gap, Better for: Filling larger gaps (3yr to 10yr) or creating smooth curves for charts. Why not forward rates? Forward rate interpolation is also acceptable (equivalent to DF interpolation), Slightly more complex to implement, Advantage: Forward curve often smoother than spot curve. Real-world systems: Bloomberg uses Nelson-Siegel-Svensson (parametric fit) for official curves, But for quick calculations, linear DF interpolation is standard, Central banks (Fed, ECB) publish smoothed curves using sophisticated splines, Trading desks often use piecewise linear DFs for speed. Important: Choice of interpolation affects pricing by a few basis points (1-5bp typically). For risk management, difference is negligible. For precise valuation (structured products), use more sophisticated methods (Hermite splines, Sobol smoothing). Key takeaway: Always interpolate on discount factors or forward rates, never directly on spot rates (unless no-arbitrage not critical).",
    },
  ];
