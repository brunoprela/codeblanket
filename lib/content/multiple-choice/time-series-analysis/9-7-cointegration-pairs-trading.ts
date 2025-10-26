export const cointegrationPairsTradingMultipleChoice = [
  {
    id: 1,
    question:
      'Two stocks have correlation = 0.95. Does this imply they are cointegrated?',
    options: [
      'Yes - high correlation implies cointegration',
      'No - correlation measures returns, cointegration measures price levels',
      'Yes - if correlation > 0.90 for 1 year+',
      'Maybe - need to check if both are I(1) first',
      'No - they are mutually exclusive concepts',
    ],
    correctAnswer: 1,
    explanation:
      "No - correlation and cointegration are different concepts. Correlation measures the linear relationship between RETURNS (or changes): $\\rho = Corr(\\Delta X, \\Delta Y)$. High correlation (0.95) means returns move together but says nothing about a stable long-run relationship in LEVELS. Cointegration requires: (1) Both series are I(1) (non-stationary in levels), (2) Linear combination is I(0) (stationary), (3) Implies mean-reverting spread. Example: Two random walks can have high correlation (by chance) but are not cointegrated. Conversely: Spot and futures prices are cointegrated (linked by arbitrage) even if daily returns have moderate correlation. For pairs trading: Need cointegration, not just correlation. High correlation without cointegration → no mean reversion → no trading signal. Test: Run Engle-Granger or Johansen test on price levels, not returns. Common error: Confusing co-movement (correlation) with equilibrium relationship (cointegration).",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'In Engle-Granger test, you regress Y on X and get residuals with ADF statistic = -3.20. Standard ADF 5% critical value is -2.86, but Engle-Granger 5% critical value is -3.34. Are Y and X cointegrated at 5% significance?',
    options: [
      'Yes - ADF statistic < -2.86',
      'No - must use Engle-Granger critical value (-3.34)',
      'Yes - both critical values are exceeded',
      'Cannot determine without sample size',
      'Yes - p-value calculation differs',
    ],
    correctAnswer: 1,
    explanation:
      "No - must compare to Engle-Granger critical value. Key point: Engle-Granger uses DIFFERENT critical values than standard ADF. Reason: Residuals from cointegrating regression are estimated (not observed), creating small-sample bias. The ADF statistic on estimated residuals has different distribution. Critical values: Standard ADF (5%): -2.86, Engle-Granger (5%): -3.34 (more negative!). Here: ADF = -3.20. Compare: -3.20 > -3.34 → Cannot reject H₀ → NOT cointegrated at 5%. Would need ADF < -3.34. Why more stringent? Testing estimated residuals is an 'easier' test (residuals forced to have mean zero), so require stronger evidence. MacKinnon (1991) provides tables for Engle-Granger critical values dependent on: number of variables (k), sample size (n), inclusion of trend. Standard packages (statsmodels.tsa.stattools.coint) automatically use correct critical values. Common error: Using standard ADF critical values leads to over-rejection (finding cointegration when none exists).",
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'A pairs trading strategy uses hedge ratio β=1.5 (estimated from regression Y = α + βX). Current prices: X=$100, Y=$155. The spread is:', 
    options: [
      '$155 - $150 = $5 (Y - β×X)',
      '$155 - $100 = $55 (Y - X)',
      '$155 / $100 = 1.55 (Y / X)',
      '$5 / 1.5 = $3.33 (standardized)',
      'Cannot calculate without α (intercept)',
    ],
    correctAnswer: 0,
    explanation:
      "Spread = Y - β×X = $155 - 1.5×$100 = $155 - $150 = $5. Definition: Spread is the residual from cointegrating regression: $\\epsilon_t = Y_t - \\alpha - \\beta X_t$. If intercept α≈0 (common after demeaning), spread ≈ Y - βX. Here: β=1.5 is the hedge ratio. Interpretation: Hold $1 of Y, short $1.5 of X → total position is the spread. Current spread = $5. Trading rules: Mean spread (from history) ≈ $0, Std(spread) ≈ $2. Z-score = ($5 - $0) / $2 = 2.5 → Sell spread (sell Y, buy X). Why not Y - X? That assumes β=1 (not estimated hedge ratio). Why not Y/X? Ratio is not stationary even if Y and X are cointegrated (multiplicative not additive). The spread MUST be stationary for pairs trading to work. Monitor spread using: rolling mean, rolling std, z-score. Entry: |z| > 2, Exit: |z| < 0.5. Position sizing: Proportional to 1/z (stronger signal = larger position).",
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      'Johansen test on 3 stocks (A, B, C) finds 2 cointegrating vectors. What does this mean for pairs trading?',
    options: [
      'Can trade 2 pairs: (A,B) and (A,C)',
      'All 3 stocks cointegrated - trade any pair',
      'Two independent mean-reverting combinations exist',
      'Need to drop one stock (only 2 are truly cointegrated)',
      'Result is inconsistent - rerun test',
    ],
    correctAnswer: 2,
    explanation:
      "Two independent mean-reverting combinations (portfolios) exist. Johansen test with 3 variables: Maximum 2 cointegrating relationships (for N variables, max N-1). Finding r=2 means: There are 2 linearly independent combinations of (A,B,C) that are stationary. These are NOT pairwise relationships! Example eigenvectors: $v_1 = [1, -0.5, -0.5]$ → Portfolio: 1×A - 0.5×B - 0.5×C, $v_2 = [0, 1, -1]$ → Portfolio: B - C. Trading implications: (1) Can construct 2 mean-reverting portfolios, (2) Not simple pairs - may involve all 3 stocks, (3) Pairs trading framework extends to 'triplets' or 'baskets'. If wanted true pairs: Could test each pair separately with Engle-Granger, but Johansen more powerful for multiple securities. Common strategy: Use first eigenvector (largest eigenvalue) as primary trading signal. Advanced: Trade both portfolios simultaneously with risk management across both. Practical consideration: More complex than pairs → higher implementation risk, but potentially more alpha if genuinely cointegrated system.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'A pairs trading strategy has Sharpe ratio = 1.8 but maximum drawdown = -45%. What does this suggest?',
    options: [
      'Excellent strategy - high Sharpe means good risk-adjusted returns',
      'Strategy likely has tail risk / rare large losses',
      'Sharpe ratio is mis-calculated',
      'Strategy is not market-neutral',
      'Need higher entry thresholds',
    ],
    correctAnswer: 1,
    explanation:
      "Strategy has tail risk - rare but severe losses. Sharpe ratio = Mean/Std = 1.8 is very good (> 1 is excellent). BUT: Max drawdown = -45% is alarming! This mismatch suggests: (1) Returns are NOT normally distributed, (2) Fat tails / negative skewness (rare large losses), (3) Strategy works most of the time but occasionally blows up, (4) Sharpe ratio understates risk (only measures volatility, not tail risk). Common causes in pairs trading: Cointegration breaks down during market stress (2008, 2020) → spread doesn't revert → losses accumulate. Stocks become correlated in crisis → all pairs lose simultaneously. Forced liquidation / margin calls → exit at worst prices. Example: LTCM (1998) - high Sharpe historically, then -90% loss when spreads widened beyond models. Risk management fixes: (1) Stop-loss at 2-3× normal spread std dev, (2) Reduce position size during high volatility, (3) Diversify across uncorrelated pairs / strategies, (4) Monitor cointegration stability (if EG test p-value deteriorates, reduce exposure). Better metrics: Sortino ratio (downside deviation), CVaR/ES (tail risk), Calmar ratio (return/max drawdown). For pairs trading: Target Sharpe 1-2 with max DD < 20%.",
    difficulty: 'advanced',
  },
];

