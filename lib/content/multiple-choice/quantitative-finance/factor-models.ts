import type { MultipleChoiceQuestion } from '@/lib/content/types';

export const factorModelsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-mc-1',
    question:
      'A regression of Portfolio X on Fama-French 6 factors yields: α=0.3% (t=0.5), β_Market=1.2, β_SMB=0.8, β_HML=-0.4, β_RMW=0.2, β_CMA=-0.3, β_MOM=0.9, R²=0.88. What does this indicate about the portfolio?',
    options: [
      'The portfolio generates significant alpha (0.3%) through stock selection skill',
      'The portfolio is a large-cap value fund with positive alpha',
      'The portfolio is a small-cap growth momentum fund with no significant alpha',
      'The portfolio replicates the market index with minimal tracking error',
    ],
    correctAnswer: 2,
    explanation:
      "Factor loadings reveal: β_SMB=0.8 (small-cap tilt), β_HML=-0.4 (negative value = growth tilt), β_MOM=0.9 (strong momentum exposure). Combined: small-cap growth with momentum → characteristic of growth + momentum strategy. Alpha: 0.3% with t=0.5 is NOT statistically significant (need |t|>1.96 for 5% significance). t=0.5 means p-value ~0.6 → 60% chance alpha is due to random chance, not skill. R²=0.88 means 88% of returns explained by factors → highly replicable. Conclusion: NO significant alpha. Portfolio is systematically exposed to small, growth, momentum factors (likely achievable through factor ETFs). Option A wrong: Alpha is insignificant. Option B wrong: Negative HML means growth (not value), and alpha is insignificant. Option D wrong: β_Market=1.2 and significant factor tilts mean it's not an index fund (index would have β≈1.0 for market, ~0 for other factors).",
  },
  {
    id: 'fm-mc-2',
    question:
      'The Fama-French Three-Factor Model shows that SMB (Size) and HML (Value) factors have historically delivered positive premia. What is the PRIMARY economic rationale for the value premium (HML)?',
    options: [
      'Value stocks are less liquid and harder to trade, requiring a liquidity premium',
      'Value stocks have higher distress risk during economic downturns, and investors demand compensation for this systematic risk',
      'Value stocks are smaller companies with less analyst coverage and information asymmetry',
      'Value stocks have higher dividend yields, providing direct cash return to investors',
    ],
    correctAnswer: 1,
    explanation:
      'The value premium (HML = High book-to-market minus Low book-to-market) is primarily explained by DISTRESS RISK. Value stocks are typically companies with: (1) Poor recent performance (low prices → high book-to-market), (2) Financial difficulties or declining businesses (higher probability of bankruptcy), (3) Cyclical sensitivity (underperform in recessions, outperform in recoveries). RISK-BASED EXPLANATION: Fama-French argue value premium is compensation for bearing systematic distress risk. During economic downturns (2008, 2020), value stocks (banks, energy, industrials) crash harder than growth stocks (tech). Investors demand higher expected return to hold riskier value stocks. BEHAVIORAL EXPLANATION (alternative): Investors overextrapolate growth for glamour stocks → overpay for growth, underpay for value. Mean reversion: Value stocks eventually revert to fair value → positive returns. Option A wrong: Liquidity premium exists but is captured more by SIZE factor (small stocks less liquid), not value specifically. Large-cap value stocks (e.g., banks) can be very liquid. Option C wrong: This describes size premium (SMB), not value. Small companies have less coverage, but value stocks can be large (e.g., Citigroup in 2009). Option D wrong: Dividend yield is correlated with value but not the causal mechanism. Many value stocks (distressed companies) cut dividends. Conversely, some growth stocks now pay dividends (Apple, Microsoft).',
  },
  {
    id: 'fm-mc-3',
    question:
      "A portfolio manager constructs a market-neutral factor strategy: Long $100M in high-momentum stocks (β_Market=1.4), Short $100M in low-momentum stocks (β_Market=1.1). What is the portfolio's market exposure, and what does this mean for returns?",
    options: [
      'Market beta = 0 (perfectly hedged); returns depend only on momentum factor',
      'Market beta = 0.3 (net long market); returns have both momentum and market exposure',
      'Market beta = 1.0 (full market exposure); returns track the market index',
      'Market beta = 2.5 (leveraged market exposure); returns amplify market moves',
    ],
    correctAnswer: 1,
    explanation:
      "Portfolio beta calculation: Long position: $100M × β_Long = $100M × 1.4 = +$140M market exposure. Short position: $100M × β_Short = $100M × 1.1 = +$110M market exposure (short reduces exposure). Net market exposure: +$140M - $110M = +$30M. Portfolio market beta: $30M / $100M capital = 0.3. INTERPRETATION: Portfolio is NOT market-neutral (β ≠ 0). It has RESIDUAL LONG MARKET BIAS (β=0.3). This is common mistake in factor investing: Assuming long-short = market neutral. Reality: If long and short legs have different betas, net exposure remains. IMPLICATIONS FOR RETURNS: Returns = α + 0.3 × (Market return) + 1.0 × (Momentum factor) + ε. Portfolio will partially track market (30% exposure). If market rallies 10%, portfolio gains ~3% from market beta (plus momentum factor return). To achieve TRUE market-neutral (β=0): Need to short more (larger notional) of low-momentum stocks, OR match dollar exposures weighted by beta. CORRECTED APPROACH: Target net beta = 0. Long $100M high-momentum (β=1.4) → +$140M market exposure. To neutralize: Short X × 1.1 = $140M → X = $127.3M short. Result: Long $100M, Short $127.3M → Net capital = -$27.3M (borrowing required or use leverage). Option A wrong: Beta is not zero (it's 0.3). Option C wrong: Beta is not 1.0 (would mean full market tracking). Option D wrong: Beta is not 2.5 (that would mean massive leverage, which is not the case here).",
  },
  {
    id: 'fm-mc-4',
    question:
      'In the Carhart Four-Factor Model, the momentum factor (MOM/UMD) has the highest historical premium (~6-8% annually) but also experienced severe crashes (e.g., -75% in 2009). What is the PRIMARY driver of momentum crashes?',
    options: [
      'Momentum strategies become overcrowded, causing simultaneous unwinding and liquidity crises',
      'During sharp market reversals, "loser" stocks rebound more strongly than "winner" stocks, causing long winners/short losers to lose on both legs',
      'Central bank interest rate hikes reduce liquidity, forcing momentum strategies to deleverage',
      'Factor correlation increases during crises, eliminating diversification benefits',
    ],
    correctAnswer: 1,
    explanation:
      "Momentum crash mechanism (2009 example): Momentum strategy: Long past winners (e.g., defensive stocks, gold, utilities that held up in 2008), Short past losers (e.g., banks, autos, cyclicals that crashed in 2008). March 2009 reversal: Market bottomed and rallied violently (+40% in 2 months). LOSER stocks (banks, cyclicals) rebounded explosively (+100-200% in months) - these were heavily shorted. WINNER stocks (defensives, gold) lagged or fell (profit-taking, rotation out of safety). Result: Short leg (losers) EXPLODES upward → massive losses. Long leg (winners) FALLS or lags → losses on both sides! Example: Short Citigroup (loser in 2008): Citi went from $1 to $4 in weeks (+300%!) → catastrophic short loss. Long utilities (winner in 2008): Utilities flat or down 10% (investors rotate to cyclicals). Net: -300% on short, -10% on long = -310% loss on position! Momentum lost ~75% in Q2 2009. ROOT CAUSE: Mean reversion during panic recoveries. Extreme losers become extremely cheap → violent bounce when fear subsides. Winners become expensive → profit-taking. This creates NEGATIVE MOMENTUM (reversal) in short term. Option A partially true: Crowding exacerbates, but not primary cause. Momentum crashed in 2009 even for uncrowded strategies. Option C wrong: 2009 crash happened WITH interest rate cuts (Fed cutting to zero), not hikes. Option D wrong: While correlations spike in crises, the specific mechanism is loser outperformance, not just correlation increase. Momentum's asymmetric payoff (losses on both legs during reversals) is unique to this factor.",
  },
  {
    id: 'fm-mc-5',
    question:
      'A factor attribution analysis shows Portfolio A has R²=0.95 and alpha=0.2% (t=1.1), while Portfolio B has R²=0.60 and alpha=1.5% (t=2.5). Both have similar total returns (12% annually). Which portfolio is MORE valuable for investors?',
    options: [
      'Portfolio A, because 95% variance explained means lower risk and more predictable returns',
      'Portfolio B, because lower R² (60%) means more idiosyncratic bets and diversification benefit',
      'Portfolio B, because it delivers significant alpha (1.5%, t=2.5) that cannot be replicated with factor ETFs',
      'Portfolio A, because insignificant alpha means lower fees and simpler replication strategy',
    ],
    correctAnswer: 2,
    explanation:
      "KEY INSIGHT: R² tells us what percentage of returns are explained by FACTORS vs ALPHA (stock-picking). Portfolio A: R²=0.95 → 95% explained by factors (market, size, value, momentum, etc.), only 5% from alpha/stock-picking. Alpha=0.2% with t=1.1 is NOT statistically significant (need t>1.96 for 5% significance). Conclusion: Portfolio A is a \"closet factor investor.\" Its returns are almost entirely replicable using factor ETFs at low cost (0.1-0.3% fees). If A charges 1% management fee but delivers 0.2% insignificant alpha, it's destroying value (fee > alpha). Portfolio B: R²=0.60 → only 60% explained by factors, 40% from alpha/stock-picking/idiosyncratic bets. Alpha=1.5% with t=2.5 IS statistically significant (t>1.96 → reject null that alpha=0 at 5% level). Conclusion: Portfolio B has genuine stock-picking skill (40% of returns are NOT factor-driven). This alpha cannot be replicated with factor ETFs → valuable unique source of return. VALUE PROPOSITION: Portfolio A: Replicable with low-cost factor ETFs (why pay high fees?). Even if same total return as B, it's \"fake alpha\" (just factor beta dressed up). Portfolio B: Delivers 1.5% true alpha (significant). If B charges 1% fee, net alpha = 0.5% (still value-add). Lower R² means LESS correlated with factor strategies → better diversifier for multi-manager portfolio. Example: If you already have factor ETF exposure, adding Portfolio A adds little (95% overlap). Adding Portfolio B adds meaningful diversification (only 60% overlap). INVESTOR PREFERENCE: Sophisticated investors: Prefer Portfolio B (true alpha, diversification, willing to pay for skill). Cost-conscious investors: Avoid Portfolio A (replicable at low cost), accept Portfolio B if fee < alpha. Option A wrong: High R² is not inherently better. It means returns are explained by factors (not unique). Option B wrong: Lower R² alone isn't valuable (could be noise). What matters is SIGNIFICANT ALPHA (which B has). Option D wrong: While A could be replicated cheaper, the question is which is more valuable - B's significant alpha is more valuable than A's replicable factor exposure.",
  },
];
