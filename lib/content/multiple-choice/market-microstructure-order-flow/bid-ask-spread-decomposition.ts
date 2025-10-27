import { MultipleChoiceQuestion } from '@/lib/types';

export const bidAskSpreadDecompositionMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'basd-mc-1',
      question:
        'Glosten-Harris regression yields: θ (adverse selection) = $0.015, ψ (order processing) = $0.005. What is the effective spread and adverse selection percentage?',
      options: [
        'Spread $0.040 (4 cents), Adverse selection 75%',
        'Spread $0.020 (2 cents), Adverse selection 50%',
        'Spread $0.010 (1 cent), Adverse selection 100%',
        'Spread $0.030 (3 cents), Adverse selection 33%',
      ],
      correctAnswer: 0,
      explanation:
        "Spread $0.040, Adverse selection 75%. Glosten-Harris model: ΔP = θ·Q + ψ·ΔQ, θ = $0.015 (permanent, adverse selection), ψ = $0.005 (transitory, order processing). Effective spread: Formula: Spread = 2 × (θ + ψ), Reason: Round-trip cost (buy then sell), Calculation: 2 × (0.015 + 0.005) = 2 × 0.020 = $0.040, Interpretation: Trader pays 4 cents to buy and immediately sell. Component breakdown: Total impact: θ + ψ = 0.015 + 0.005 = $0.020 per trade, Adverse selection: θ = $0.015 (75% of 0.020), Order processing: ψ = $0.005 (25% of 0.020), Percentage: 0.015 / 0.020 = 0.75 = 75% adverse selection. Why 75% adverse selection? Calculation: θ / (θ + ψ) = 0.015 / 0.020 = 0.75, Meaning: 75% of spread due to information asymmetry, High: Indicates significant informed trading, Market maker: Loses money on 75% of spread (permanent price moves against them). Implications: Market maker response: Widen spread to compensate (high adverse selection), Quote size: Reduce size (limit exposure to informed traders), Execution quality: Worse for uninformed traders (subsidize informed traders' profits). Why not $0.020? Missing factor of 2: Effective spread = 2 × (θ + ψ), not just (θ + ψ), Round-trip: Need to account for both buy and sell sides, Calculation error: 0.015 + 0.005 = 0.020, but spread = 2 × 0.020 = 0.040. Why not 50% adverse selection? 50% would mean: θ = ψ (equal components), Our case: θ = $0.015 > ψ = $0.005 (θ is 3× larger), Ratio: 0.015 / 0.020 = 75%, not 50%. Practical context: Typical values: Adverse selection: 30-60% for liquid stocks, >70% for illiquid or volatile stocks, Our case: 75% is high (significant informed trading). Comparison: Low adverse selection (30%): θ = $0.003, ψ = $0.007, spread = $0.020, Mostly uninformed flow (safe for market makers), High adverse selection (75%): θ = $0.015, ψ = $0.005, spread = $0.040, Mostly informed flow (dangerous for market makers).",
    },
    {
      id: 'basd-mc-2',
      question:
        'Roll model estimates spread at $0.008 but quoted spread is $0.010. Why does Roll underestimate?',
      options: [
        'Roll ignores adverse selection (assumes no informed trading)',
        'Roll overestimates spread (includes too much noise)',
        'Roll and quoted should always match (data error)',
        'Roll measures different spread (effective vs quoted)',
      ],
      correctAnswer: 0,
      explanation:
        'Roll ignores adverse selection. Roll model assumptions: Bid-ask bounce: Trades alternate between bid and ask, No information: Price changes purely due to liquidity provision (no informed trading), Negative serial covariance: Caused by bounce (up then down). Roll formula: S = 2√(-Cov(ΔP_t, ΔP_{t-1})), Measures: Order processing cost only (transaction bounce), Ignores: Adverse selection (informed trading permanent impact). Spread decomposition: Total spread: Processing + Adverse selection = 0.010, Roll estimate: Processing only ≈ 0.008, Missing: Adverse selection ≈ 0.002 (0.010 - 0.008). Calculation: If adverse selection = 20% of total spread, Roll = 0.80 × Quoted = 0.80 × 0.010 = 0.008, Matches our observation (Roll underestimates by ~20%). Why underestimate? Informed trading: Some trades move price permanently (not just bounce), Roll assumption: Violated (not all trades are uninformed), Result: Negative serial covariance smaller in magnitude (less bounce), Spread estimate: Lower (sqrt of smaller covariance). Example: Pure bounce (Roll assumption): Prices: 100.00, 100.01, 100.00, 100.01 (perfect alternation), Serial cov: -0.0001 (highly negative), Roll spread: Large (captures full bounce). With informed trading: Prices: 100.00, 100.01, 100.02, 100.03 (trend + bounce), Serial cov: -0.00005 (less negative, trend reduces bounce), Roll spread: Smaller (underestimates true spread). Why not overestimate? Overestimate would mean: Roll > Quoted, Requires: Stronger negative serial covariance than actual, Reality: Informed trading reduces serial covariance (trend counteracts bounce), Result: Roll < Quoted (underestimate, not overestimate). Why not always match? Different concepts: Roll: Effective spread (realized transaction cost from bounce), Quoted: Posted spread (bid-ask difference at point in time), Can differ: Especially with informed trading, price improvement, or hidden liquidity. Why not effective vs quoted? Effective spread: Realized cost (price relative to mid), Quoted spread: Posted bid-ask, Roll measures: Effective spread estimate (from price series), Still underestimates: Because ignores adverse selection component. Validation: Glosten-Harris: Decomposes spread into processing + adverse selection, Roll: Estimates processing only (missing adverse selection), Difference: Quoted - Roll ≈ Adverse selection component. Real-world: Liquid stocks: Roll ≈ 0.8-0.9 × Quoted (low adverse selection, 10-20%), Illiquid stocks: Roll ≈ 0.5-0.7 × Quoted (high adverse selection, 30-50%), Volatile periods: Roll much lower (informed trading increases).',
    },
    {
      id: 'basd-mc-3',
      question:
        'Kyle lambda = 0.0003 means a 10,000 share buy order causes what price impact?',
      options: [
        '$3.00 (10,000 × 0.0003)',
        '$0.30 (sqrt(10,000) × 0.0003)',
        '$0.003 (10,000 / 10,000 × 0.0003)',
        '$30.00 (10,000 × 0.0003 × 10)',
      ],
      correctAnswer: 0,
      explanation:
        "Impact = $3.00. Kyle lambda definition: λ = price impact per unit volume, Formula: ΔP = λ × Q, where ΔP = price change, Q = signed volume (+buy, -sell). Calculation: λ = 0.0003, Q = 10,000 shares (buy), Impact: ΔP = 0.0003 × 10,000 = 3.00, Interpretation: Price rises $3.00 due to buy order. Linear relationship: Doubling order: 10K → 20K shares, doubles impact: $3 → $6, Kyle model: Linear impact (contrast with square-root law which is non-linear). Components: λ measures: Adverse selection (how informed is flow), Higher λ: More informed trading (larger price impact per share), Lower λ: Less informed trading (smaller price impact). Example: Before order: Price $100.00, After 10K buy: Price $103.00 (moved up $3), Cause: Market infers information from large buy (bullish signal). Why not $0.30? Would require: sqrt(10,000) × λ = 100 × 0.0003 = 0.03 (not 0.30), Confusion: Mixing Kyle lambda (linear) with square-root law (non-linear), Kyle: Linear impact (price proportional to quantity), Square-root: Sub-linear (price proportional to sqrt(quantity)). Why not $0.003? Would require: Q = 10 shares (not 10,000), or λ = 0.0000003 (not 0.0003), Calculation error: 10,000 × 0.0003 = 3.00 (not 0.003). Kyle vs Square-root: Kyle (linear): Impact = λ × Q, used for adverse selection measurement, Assumed: Informed trader (price moves with order), Square-root (sub-linear): Impact = γ × σ × sqrt(Q/V), used for market impact estimation, Assumed: Uninformed trader (temporary liquidity effect). When to use Kyle: Short-term: Immediate price impact, Information: Measuring informed trading intensity, Permanent: Impact doesn't revert (information incorporated). When to use Square-root: Execution: Minimizing market impact over time, Liquidity: Large orders broken into pieces, Temporary: Partial reversion after execution. Real-world values: Liquid stocks: λ ≈ 0.00001 - 0.0001 (1-10 bps per 1000 shares), Illiquid stocks: λ ≈ 0.0001 - 0.001 (10-100 bps per 1000 shares), Our example: λ = 0.0003 (30 bps per 1000 shares, moderately liquid). Trading implications: Cost: 10K share order costs $3 in price impact, Total cost: If stock is $100, 10K × $100 = $1M notional, Impact: $3 / $1M = 0.0003 = 3 bps (0.03%), Compare: Commission ~$10 (1 bp), spread ~$100 (10 bps), impact $3000 (300 bps if counting slippage on full position). Correction: If $100 stock, $3 move = 3% (not 3 bps), $3 impact on 10K shares = $30K total cost, Per share: $3 / 10,000 = $0.0003 = 0.3 bps per share, Total position: $3 on $1M = 0.3% = 30 bps.",
    },
    {
      id: 'basd-mc-4',
      question:
        'A stock has serial covariance of returns Cov(ΔP_t, ΔP_{t-1}) = +0.0001 (positive). What does this indicate for the Roll model?',
      options: [
        'Roll model fails (assumes negative covariance, positive suggests momentum/informed trading)',
        'Roll estimates spread as 2√(0.0001) = $0.02',
        'Covariance should be negative (data error, recompute)',
        'Roll model works better (positive covariance improves accuracy)',
      ],
      correctAnswer: 0,
      explanation:
        'Roll fails (positive covariance). Roll assumption: Negative serial covariance: Cov(ΔP_t, ΔP_{t-1}) < 0, Reason: Bid-ask bounce (price alternates up-down), Formula: S = 2√(-Cov), requires Cov < 0 (to take sqrt of positive number). Positive covariance: Observation: Cov = +0.0001 > 0, Interpretation: Momentum (price continues in same direction), Indicates: Informed trading (traders push price in one direction), Contradicts: Roll assumption (no information). Roll model failure: Mathematical: Cannot compute sqrt(-Cov) = sqrt(-0.0001) = undefined (sqrt of negative), Return: NaN or 0 (signal model failure), Alternative: Use other estimators (Glosten-Harris, Kyle lambda). Why positive covariance? Informed buying: Day 1: Large buy → price up $0.01, Day 2: More buying (following info) → price up $0.01, Serial cov: Positive (both changes same sign), Momentum: Returns positively autocorrelated. Market microstructure: Price discovery: Information takes time to be incorporated, Gradual impact: Large order split over time (each piece pushes price up), Result: Positive serial correlation (trend continues). Why not estimate as $0.02? Cannot take sqrt: 2√(+0.0001) requires sqrt of positive cov, Roll formula: S = 2√(-Cov), needs negative, Math error: sqrt(-0.0001) is imaginary number (not real spread). Correct interpretation: Positive cov: Violates Roll assumption (model inapplicable), Need different model: Glosten-Harris (handles information), Kyle lambda (measures adverse selection). Why not data error? Positive covariance: Can be real phenomenon (informed trading), Not necessarily error: Momentum and informed trading create positive autocorrelation, Check data: Verify, but positive cov is economically meaningful. When does Roll work? Negative covariance: Uninformed flow (alternating bid-ask), Liquid stocks: High-frequency trading (quote-to-quote), Calm markets: Low information events, Example: Cov = -0.0001 → Spread = 2√(0.0001) = $0.02 (valid). When does Roll fail? Positive covariance: Informed trading (momentum), Trending markets: Persistent price moves, News events: Earnings, economic releases (directional flow), Example: Cov = +0.0001 → Roll undefined (model fails). Alternative estimators: Glosten-Harris: ΔP = θ·Q + ψ·ΔQ (regression), handles informed trading, Kyle lambda: λ = Cov(ΔP, Q) / Var(Q), measures adverse selection directly, Huang-Stoll: Three components (processing, inventory, adverse selection). Real-world frequency: Roll fails: 10-20% of the time (periods of informed trading, momentum), Glosten-Harris: More robust (works 95%+ of time), Kyle: Always works (designed for informed trading).',
    },
    {
      id: 'basd-mc-5',
      question:
        'Market maker has adverse selection component θ = $0.020 and order processing ψ = $0.010. Informed trading increases θ to $0.030. How should spread change?',
      options: [
        'Widen from $0.060 to $0.080 (increase by $0.020)',
        'Widen from $0.030 to $0.040 (increase by $0.010)',
        'Stay at $0.060 (order processing unchanged)',
        'Widen from $0.020 to $0.030 (match theta increase)',
      ],
      correctAnswer: 0,
      explanation:
        'Widen from $0.060 to $0.080. Initial spread: Components: θ = $0.020 (adverse selection), ψ = $0.010 (order processing), Spread: 2 × (θ + ψ) = 2 × (0.020 + 0.010) = 2 × 0.030 = $0.060. After informed trading increase: New θ = $0.030 (increased from $0.020, +$0.010), ψ unchanged = $0.010 (order processing constant), New spread: 2 × (0.030 + 0.010) = 2 × 0.040 = $0.080, Change: $0.080 - $0.060 = $0.020 increase. Why widen by $0.020? Adverse selection up: θ increased by $0.010, Round-trip: Spread accounts for both sides (× 2), Total: 2 × $0.010 = $0.020 spread increase, Reason: Market maker needs to compensate for higher information risk. Market maker logic: Before (θ = $0.020): 66% adverse selection (0.020 / 0.030), Moderate informed trading, Spread: $0.060 (manageable). After (θ = $0.030): 75% adverse selection (0.030 / 0.040), High informed trading, Spread: $0.080 (protect from losses). Why not $0.030 to $0.040? Incorrect base: Initial spread is $0.060 (not $0.030), $0.030 is: Half the spread (one side), or the component sum (before factor of 2), Actual spread: Always 2 × (θ + ψ). Why not stay at $0.060? Adverse selection increased: θ went from $0.020 to $0.030, Market maker: Faces higher risk (more informed traders), Must adjust: Widen spread to compensate for losses, Static spread: Would lead to negative expected profits (lose to informed). Why not match theta ($0.020 to $0.030)? Theta increase: $0.010 (component-level), Spread increase: $0.020 (effective spread-level), Factor of 2: Spread = 2 × components (buy and sell sides), Correct: Spread moves 2× the component move. Economic intuition: Informed trading: More traders with private information, Market maker risk: Buys high (before price drops), sells low (before price rises), Compensation: Needs wider spread to break even, Expected loss: Higher θ → more adverse selection → widen spread. Real-world example: Normal conditions: θ = $0.020, spread = $0.060, Bid $100.00, Ask $100.06, News event (earnings): θ spikes to $0.030, spread widens to $0.080, Bid $100.00, Ask $100.08, Return to normal: θ drops back to $0.020, spread tightens to $0.060. Alternative response: Instead of widening spread: Reduce quote size (500 → 100 shares), Increase cancel rate (refresh quotes faster), Stop quoting: Temporarily exit market (extreme), But if still quoting: Must widen spread (compensate for risk).',
    },
  ];
