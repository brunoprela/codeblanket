export const financialMathPuzzlesQuiz = [
  {
    id: 'fmp-q-1',
    question:
      'Jane Street: "A bond has 10 years to maturity, 5% annual coupon, face value $1000, currently trading at $920. Calculate: (1) current yield, (2) approximate YTM using the approximation formula, (3) duration approximately (assume YTM ≈ 6%), (4) if interest rates increase by 1%, estimate new bond price using duration. Show all calculations and explain why duration approximation may be inaccurate for large rate changes."',
    sampleAnswer:
      'Complete bond analysis: (1) Current yield: Annual coupon / Price = 50 / 920 ≈ 5.43%. This measures income return only, ignoring capital gain to par. (2) YTM approximation: Formula YTM ≈ [C + (F-P)/n] / [(F+P)/2] where C=50, F=1000, P=920, n=10. YTM ≈ [50 + (1000-920)/10] / [(1000+920)/2] = [50 + 8] / 960 = 58/960 ≈ 6.04%. This captures both income and capital appreciation. (3) Duration approximation: For bond with n years, coupon c, yield y: Modified duration ≈ [1 - (1+y)^(-n)] / y. For y=0.06, n=10: Duration ≈ [1 - (1.06)^(-10)] / 0.06 = [1 - 0.558] / 0.06 = 0.442 / 0.06 ≈ 7.37 years. More precisely, Macaulay duration ≈ [(1+y) - (1+y+n (c-y))/c((1+y)^n-1)+y)] / y. For c=0.05, y=0.06, n=10: Duration ≈ 7.8 years. Modified duration = Macaulay / (1+y) ≈ 7.8/1.06 ≈ 7.36. (4) Price change estimate: ΔP/P ≈ -Duration × Δy. For Δy = +0.01 (1% increase): ΔP/P ≈ -7.36 × 0.01 = -0.0736 = -7.36%. New price ≈ 920 × (1 - 0.0736) = 920 × 0.9264 ≈ $852. Verification: At y=7%, price = Σ(50/(1.07)^t) + 1000/(1.07)^10. This requires calculation but approximately $860. Our duration estimate of $852 is close. (5) Why duration approximation inaccurate: Duration is first-order (linear) approximation. For large rate changes, convexity (second-order effect) matters. Bonds have positive convexity: actual price change is less negative than duration predicts for rate increases (good for bondholder). For 1% change, error is moderate (~1%). For 5% change, error could be 10%+. Interview: show YTM approximation, estimate duration mentally (~7-8 years for 10-year bond), apply -Duration×Δy rule, acknowledge convexity for precision.',
    keyPoints: [
      'Current yield: 50/920 = 5.43% (income only)',
      'YTM approximation: [50+8]/960 ≈ 6.04% (income + capital gain)',
      'Duration ≈ 7.4 years for 10-year bond at 6% yield',
      'Price change: -Duration×Δy = -7.4×1% = -7.4% → price falls to ~$852',
      'Duration is linear approximation; convexity needed for large changes',
    ],
  },
  {
    id: 'fmp-q-2',
    question:
      'Citadel: "You have three currency pairs: EUR/USD = 1.10, USD/JPY = 110, EUR/JPY = 120. Is there an arbitrage opportunity? If so, describe the exact trades to capture it and calculate the profit per $1 million invested. Then explain how bid-ask spreads affect arbitrage and estimate the minimum spread width that would eliminate the opportunity."',
    sampleAnswer:
      'Complete currency arbitrage analysis: (1) Check triangular arbitrage: Cross-rate should be EUR/JPY = (EUR/USD) × (USD/JPY) = 1.10 × 110 = 121. Actual rate: EUR/JPY = 120. Discrepancy: 121 vs 120 (EUR overvalued vs JPY in direct market). (2) Arbitrage strategy: Start with $1M USD. Route A (via EUR/JPY direct): $1M → €909,091 (divide by 1.10) → ¥109,090,910 (multiply by 120). Route B (via triangular): $1M → ¥110,000,000 (multiply by 110). Route B yields more yen! Execute: $1M → ¥110M → €909,091 (divide by 121 cross-rate if using implied) → ... Wait, I need to reconsider which direction to trade. Let me recalculate. Implied EUR/JPY = 121, actual = 120. Yen is expensive (fewer yen per euro in actual market). Arbitrage: Buy yen cheap via USD/JPY, sell yen expensive via EUR/JPY. Trade sequence: (a) $1M USD → ¥110M (at USD/JPY = 110), (b) ¥110M → €916,667 (at EUR/JPY = 120, so 1 EUR = 120 JPY, thus ¥110M / 120 = €916,667), (c) €916,667 → $1,008,333 (at EUR/USD = 1.10, so 1 EUR = 1.10 USD). Profit: $1,008,333 - $1,000,000 = $8,333 per $1M (0.833% return). (3) Bid-ask spread impact: Let s = half-spread. Transactions incur 2s cost per leg (buy at ask, sell at bid). Three legs → total cost ≈ 6s. For arbitrage to exist: Profit 0.833% > Cost 6s. Minimum spread to eliminate: 6s ≈ 0.833% → s ≈ 0.139% = 14 bps. If each currency pair has bid-ask spread > 14 bps, arbitrage disappears (typical FX spreads for majors are 1-5 bps for size, so this opportunity would exist in real markets temporarily before being arbitraged away). Interview: identify discrepancy (121 vs 120), execute triangular trade capturing the difference, calculate precise profit, acknowledge transaction costs eliminate small inefficiencies.',
    keyPoints: [
      'Implied EUR/JPY: 1.10 × 110 = 121, actual = 120 (discrepancy)',
      'Trade: $1M → ¥110M → €916,667 → $1,008,333',
      'Profit: $8,333 per $1M (0.833% return)',
      'Bid-ask spreads: 3 legs × 2 sides → 6s total cost',
      'Break-even spread: 0.833% / 6 ≈ 14 bps per currency pair',
    ],
  },
  {
    id: 'fmp-q-3',
    question:
      'Two Sigma: "A stock currently at $100 pays no dividends. The risk-free rate is 2%. A 1-year forward contract on the stock should trade at what price? Now suppose you observe the forward trading at $105. Describe the arbitrage strategy including: (1) exact positions to take today, (2) cash flows today and in 1 year, (3) profit locked in, (4) what happens if stock ends at $80, $110, or $130. Why is this arbitrage risk-free regardless of stock price?"',
    sampleAnswer:
      "Complete forward arbitrage analysis: (1) Theoretical forward price: F = S × e^(rT) = 100 × e^(0.02×1) = 100 × 1.0202 ≈ $102.02. No dividends so q=0. (2) Observed price: $105 > $102.02. Forward is overpriced by ~$2.98. (3) Arbitrage strategy: Today (t=0): (a) Borrow $100 at 2% risk-free rate, (b) Buy stock for $100 (using borrowed money), (c) Sell forward contract at $105 (agree to sell stock in 1 year for $105). Net cash flow today: -$100 (buy stock) + $100 (borrow) = $0. In 1 year (t=1): (a) Deliver stock via forward → receive $105, (b) Repay loan: $100 × e^0.02 = $102.02. Net cash flow: +$105 - $102.02 = +$2.98 profit (risk-free!). (4) Scenarios at expiration: Stock at $80: Forward contract forces you to sell at $105 (above market $80, but you're okay because you own the stock). Profit = $105 - $102.02 = $2.98 ✓. Stock at $110: Forward forces sale at $105 (below market $110, but you still profit). Profit = $105 - $102.02 = $2.98 ✓. Stock at $130: Same analysis. Profit = $2.98 ✓. (5) Why risk-free: You locked in both sides at t=0: borrowed at 2%, sold forward at $105. No market risk because you own the stock (covered forward) and forward price is contractually fixed. Any stock price movement is irrelevant - you deliver what you own for the agreed $105. The profit of $2.98 is guaranteed at t=0 via arbitrage pricing violation. Interview: state theoretical price formula, identify overpricing, describe cash-and-carry arbitrage (borrow, buy spot, sell forward), show profit is locked in regardless of stock movement, emphasize risk-free nature of arbitrage.",
    keyPoints: [
      'Theoretical forward: F = 100 × e^0.02 ≈ $102.02',
      'Observed $105 → overpriced by $2.98',
      'Arbitrage: borrow $100, buy stock, sell forward at $105',
      'At expiration: deliver stock for $105, repay $102.02, profit $2.98',
      'Risk-free: locked in prices at t=0, profit independent of stock price',
    ],
  },
];
