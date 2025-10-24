import { MultipleChoiceQuestion } from '@/lib/types';

export const portfolioTheoryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pt-mc-1',
    question:
      'Two assets A and B have returns: E[R_A]=10%, E[R_B]=8%, σ_A=20%, σ_B=15%, correlation=0.3. What is the approximate portfolio volatility for 50-50 allocation?',
    options: [
      '17.5% (simple average)',
      '14.2% (diversification benefit)',
      '12.5% (maximum diversification)',
      '20% (highest individual volatility)',
    ],
    correctAnswer: 1,
    explanation:
      'Portfolio variance: σ²_p = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂ = 0.5²(0.20²) + 0.5²(0.15²) + 2(0.5)(0.5)(0.3)(0.20)(0.15) = 0.01 + 0.005625 + 0.0045 = 0.020125. σ_p = √0.020125 = 14.2%. Simple average would be 17.5%, but correlation<1 provides diversification benefit reducing risk to 14.2%. If correlation=1, no diversification (17.5%). If correlation=0, more benefit (σ≈13%).',
  },
  {
    id: 'pt-mc-2',
    question:
      'According to CAPM, Stock X has β=1.5, market return=10%, risk-free rate=3%. What is the expected return for Stock X?',
    options: [
      '10.5% (market return × beta)',
      '13.5% (Rf + β × market)',
      '13.5% (Rf + β × (Rm - Rf))',
      '15% (beta × market)',
    ],
    correctAnswer: 2,
    explanation:
      'CAPM formula: E[R] = Rf + β × (Rm - Rf). Market risk premium = 10% - 3% = 7%. E[R_X] = 3% + 1.5 × 7% = 3% + 10.5% = 13.5%. Common mistake: Using β × Rm instead of β × (Rm - Rf). Beta measures sensitivity to excess market return, not total market return. Stock with β=1.5 amplifies market risk premium by 1.5×.',
  },
  {
    id: 'pt-mc-3',
    question:
      'Portfolio A: Sharpe=0.8, Portfolio B: Sharpe=1.2. You can borrow/lend at risk-free rate. Which creates the highest Sharpe ratio?',
    options: [
      'Leverage Portfolio A 1.5× (higher risk, higher return)',
      'Portfolio B with any leverage (maintains Sharpe)',
      '50-50 mix of A and B (diversification)',
      'Portfolio A unleveraged (less risky)',
    ],
    correctAnswer: 1,
    explanation:
      'Sharpe ratio is invariant to leverage! Leveraging Portfolio B by 2× doubles both return and volatility, but Sharpe stays 1.2. Any leverage factor: Sharpe(leveraged) = [L×(R-Rf)] / [L×σ] = (R-Rf)/σ = original Sharpe. Since B has higher Sharpe (1.2 > 0.8), any leverage of B beats any leverage of A. Mixing A and B reduces Sharpe below B alone. Best: Portfolio B with optimal leverage based on risk tolerance.',
  },
  {
    id: 'pt-mc-4',
    question:
      'An asset has positive alpha (+2%) relative to CAPM. What does this indicate?',
    options: [
      'Asset is riskier than market (higher beta)',
      'Asset is overvalued (eliminate position)',
      'Asset is undervalued (generating excess returns)',
      'Asset has zero correlation with market',
    ],
    correctAnswer: 2,
    explanation:
      "Positive alpha (α>0) means actual return exceeds CAPM expected return given beta. α = R_actual - [Rf + β×(Rm-Rf)] = 2%. Interpretation: Asset outperforming what risk predicts → undervalued or superior management. Action: BUY or OVERWEIGHT. Market hasn't fully priced skill/advantage. Negative alpha (-2%) = overvalued → SELL. Alpha=0 = fairly valued. Active managers seek positive alpha assets.",
  },
  {
    id: 'pt-mc-5',
    question: 'Which portfolio lies on the efficient frontier?',
    options: [
      'Return=8%, Volatility=12% when another has Return=10%, Volatility=12%',
      'Return=10%, Volatility=15% when another has Return=10%, Volatility=20%',
      'Return=12%, Volatility=18%, maximum Sharpe ratio at this risk level',
      'Return=15%, Volatility=25%, but achievable with lower risk',
    ],
    correctAnswer: 2,
    explanation:
      'Efficient frontier contains portfolios with HIGHEST return for each risk level (or LOWEST risk for each return). Option 1: Dominated (same risk, lower return). Option 2: On frontier (lower risk, same return = efficient). Option 3: On frontier (max Sharpe at that risk). Option 4: Not efficient (same return achievable with less risk). Only portfolios that cannot be improved lie on efficient frontier.',
  },
];
