import { MultipleChoiceQuestion } from '@/lib/types';

export const riskMeasuresMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'rm-mc-1',
    question:
      'Portfolio: $10M, daily return μ=0.05%, σ=1.5%. What is 1-day 95% VaR using parametric method?',
    options: [
      'VaR = $10M × 1.5% = $150,000',
      'VaR = $10M × 1.65 × 1.5% = $247,500',
      'VaR = $10M × (0.05% + 1.65×1.5%) = $253,000',
      'VaR = $10M × (1.65×1.5% - 0.05%) = $242,500',
    ],
    correctAnswer: 3,
    explanation:
      'VaR_95% = -(μ + z×σ) where z=-1.65 for losses. Calculation: VaR = -(0.05% + (-1.65)×1.5%) = -(0.05% - 2.475%) = -(-2.425%) = 2.425%. Dollar VaR = $10M × 2.425% = $242,500. Interpretation: 95% confident daily loss will NOT exceed $242,500. Or: Expect losses > $242,500 on 5% of days (1 day per month). Note: Expected return μ slightly reduces VaR (drift component). If μ=0, VaR would be $247,500. Positive expected return provides small buffer.',
  },
  {
    id: 'rm-mc-2',
    question:
      'Historical VaR (\$320K) is 2x higher than parametric VaR (\$160K) for same portfolio. What explains this?',
    options: [
      'Calculation error; both methods should give same result',
      'Historical method double-counts volatility',
      'Historical data includes fat tails and crisis periods (kurtosis > 3)',
      'Parametric method uses wrong confidence level',
    ],
    correctAnswer: 2,
    explanation:
      'Historical VaR often significantly higher because: (1) Fat tails: Real returns have kurtosis > 3 (more extreme events than normal). Example: 2008 crisis days with -8% losses (5σ+ events under normal distribution). (2) Non-normal distribution: Parametric assumes normal (thin tails, symmetric). Historical captures actual left tail (thick, negative skew). (3) Volatility clustering: Historical includes high-vol periods (2008, 2020). Parametric uses average volatility (underestimates crisis risk). Example: Historical VaR includes 13th worst day = -3.2% loss. Parametric VaR = 1.65σ = -1.6% (assumes normal). Historical 2x higher reflects REALITY of fat tails. This is why stress testing complements VaR.',
  },
  {
    id: 'rm-mc-3',
    question:
      '10-day VaR using square-root rule: VaR(10-day) = VaR(1-day) × √10. What is the KEY assumption?',
    options: [
      'Returns are normally distributed',
      'Returns are independent and identically distributed (i.i.d.)',
      'Volatility is constant over 10 days',
      'Portfolio composition does not change',
    ],
    correctAnswer: 1,
    explanation:
      "Square-root-of-time rule assumes i.i.d. returns: Independent: Today\'s return doesn't affect tomorrow's (no autocorrelation). Identically distributed: Same distribution each day (constant volatility, no regime changes). Under i.i.d., variance scales linearly: Var(10-day) = 10 × Var(1-day). Std dev scales as square root: σ(10-day) = √10 × σ(1-day). REALITY violates i.i.d.: Volatility clustering: After -3% day, volatility often doubles (GARCH effects). √10 rule underestimates risk. Mean reversion: Extreme losses partially revert, reducing multi-day loss. Liquidity: Selling $500K over 10 days vs 1 day → different market impact. Better approach: Multi-day Monte Carlo with GARCH volatility, or overlapping historical windows. Actual 10-day VaR often 1.5-2.5x (not 3.16x) 1-day VaR.",
  },
  {
    id: 'rm-mc-4',
    question: 'CVaR (Conditional VaR) vs VaR. Which statement is TRUE?',
    options: [
      'CVaR is always equal to VaR at the same confidence level',
      'CVaR is always greater than VaR (captures tail risk beyond VaR)',
      'CVaR is non-subadditive like VaR (violates diversification)',
      'CVaR = VaR only for normal distributions',
    ],
    correctAnswer: 1,
    explanation:
      'CVaR (Expected Shortfall) = E[Loss | Loss > VaR]. CVaR always ≥ VaR because: VaR = threshold at confidence level (e.g., 95th percentile). CVaR = AVERAGE of losses beyond VaR (average of worst 5%). Example: VaR_95 = $200K (95th percentile loss). Worst 5% days: -$200K, -$220K, -$240K, -$280K, -$350K (average = -$258K). CVaR_95 = $258K > VaR (\$200K). CVaR captures tail risk that VaR ignores. KEY ADVANTAGE: CVaR is subadditive (satisfies diversification), VaR is NOT. Basel IV switched from VaR to Expected Shortfall for bank capital. CVaR used in portfolio optimization (minimize tail risk).',
  },
  {
    id: 'rm-mc-5',
    question:
      'Reverse stress testing asks: "What scenario causes portfolio failure?" Why is this useful vs VaR?',
    options: [
      'Replaces VaR (reverse stress testing is strictly better)',
      'Identifies rare but catastrophic scenarios VaR assigns negligible probability',
      'Reverse stress testing is required by Basel III regulations',
      'Reverse stress testing is easier to calculate than VaR',
    ],
    correctAnswer: 1,
    explanation:
      'Reverse stress testing complements VaR by identifying BLACK SWANS: VaR says: "Risk is $200K (2% loss) in 95% of cases." Focus: Frequent, small losses. Reverse stress asks: "What causes -50% loss (failure)?" Answer: Stocks -65% + bonds -30% simultaneously. Probability: <1% (VaR assigns ~0.0001%, considers negligible). Reality: 2008 was worse than VaR predicted (Lehman lost 100x VaR in 1 day). USE CASE: Identify existential threats even if "extremely unlikely." Example: Failure if rates rise 10% (bonds -60%, real estate -70%). VaR says: 0.00001% probability, ignore. Reverse stress: Plausible if Fed fights hyperinflation → mitigate by reducing duration NOW. Combined framework: VaR: Daily risk management. Stress testing: Quarterly vulnerability check. Reverse stress: Annual "what could kill us?" analysis. Banks required to show: No plausible scenario causes insolvency.',
  },
];
