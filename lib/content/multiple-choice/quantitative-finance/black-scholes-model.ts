import { MultipleChoiceQuestion } from '@/lib/types';

export const blackScholesModelMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bsm-mc-1',
    question:
      'Given S=$100, K=$100, T=0.25 years, r=5%, σ=30%, what is d1 in Black-Scholes formula?',
    options: [
      'd1 = [ln(100/100) + (0.05 + 0.30²/2)×0.25] / (0.30×√0.25) = 0.1625',
      'd1 = [ln(100/100) + (0.05 - 0.30²/2)×0.25] / (0.30×√0.25) = -0.0875',
      'd1 = (100-100) / (0.30×√0.25) = 0',
      'd1 = [ln(100/100) + 0.05×0.25] / 0.30 = 0.0417',
    ],
    correctAnswer: 0,
    explanation:
      'd1 = [ln(S/K) + (r + σ²/2)T] / (σ√T). Calculation: ln(100/100) = 0, (0.05 + 0.09/2)×0.25 = 0.02438, σ√T = 0.30×0.5 = 0.15. d1 = 0.02438/0.15 = 0.1625. This represents standardized distance from strike, adjusted for drift and volatility. Positive d1 means call is slightly in-the-money on expected basis.',
  },
  {
    id: 'bsm-mc-2',
    question:
      'Implied volatility is 28% for a call option. Stock suddenly jumps 5% up. What happens to implied volatility?',
    options: [
      'IV increases (higher realized volatility)',
      'IV stays at 28% (independent of stock movement)',
      'IV typically decreases due to "vol crush"',
      'IV goes to zero (option now ITM with no uncertainty)',
    ],
    correctAnswer: 2,
    explanation:
      'Counterintuitively, IV often decreases after sharp moves (especially upward moves in equities). Reasons: (1) Leverage effect: Stock up → lower leverage → lower future volatility expected. (2) Reduced uncertainty: Big move happened, less uncertainty remaining. (3) Put demand drops: Crash fear reduced after rally. (4) Vol mean reversion: After spike in realized vol, market expects return to normal. This is why "selling vol" after events can be profitable.',
  },
  {
    id: 'bsm-mc-3',
    question:
      'Why is implied volatility higher for deep OTM puts than ATM options in equity markets (volatility skew)?',
    options: [
      'Deep OTM puts are more liquid and have higher demand',
      'Black-Scholes model systematically overprices OTM puts',
      'Investor demand for downside protection and fat-tail crash risk',
      'Market makers charge higher premiums for all OTM options',
    ],
    correctAnswer: 2,
    explanation:
      'Volatility skew in equities (higher IV for OTM puts) reflects: (1) Crash fear: Post-1987, investors pay premium for tail risk protection. (2) Leverage effect: Stock drops → leverage increases → volatility increases (asymmetric). (3) Demand for puts: Portfolio insurance, hedging. (4) Fat tails: Real distribution has more extreme downside moves than lognormal. Example: SPY $450, $400 put (11% OTM) has IV=28% vs ATM IV=20%. Market pays 40% premium for crash protection.',
  },
  {
    id: 'bsm-mc-4',
    question:
      'A stock pays a 3% annual dividend yield. How does this affect call and put prices under Black-Scholes?',
    options: [
      'Increases both call and put prices (more cashflows)',
      'Decreases call prices, increases put prices',
      'Increases call prices, decreases put prices',
      'No effect (dividends only matter at payment date)',
    ],
    correctAnswer: 1,
    explanation:
      'Dividends decrease stock price by dividend amount on ex-date. Adjusted Black-Scholes: S_adj = S×e^(-qT). Lower effective stock price: Decreases call value (less likely to finish ITM), Increases put value (more likely to finish ITM). Example: S=$100, q=3%, T=1 year → S_adj=$100×e^(-0.03)=$97.04. Call worth less, put worth more. For call holders: Lose dividend, opportunity cost. For put holders: Stock drops by dividend, benefit from lower price.',
  },
  {
    id: 'bsm-mc-5',
    question:
      'If Black-Scholes call price=$5 but market price=$6, which statement is correct?',
    options: [
      'Arbitrage: Buy Black-Scholes call at $5, sell market call at $6, profit $1',
      'Implied volatility > input volatility used in Black-Scholes calculation',
      'Black-Scholes is wrong; market price is always correct',
      'Transaction costs explain the $1 difference',
    ],
    correctAnswer: 1,
    explanation:
      'Black-Scholes uses volatility input σ. If market price > BS price: Market is pricing higher volatility than your input. Implied volatility (IV from market $6) > assumed volatility (σ used to get $5). Example: You calculated with σ=20%, but market IV=25%. Increase σ to 25% in BS formula → price rises to $6. No arbitrage here (you can\'t "buy BS at $5" - BS is just a calculator). Market price reflects collective wisdom about future volatility. Use IV extraction to find what volatility market is pricing.',
  },
];
