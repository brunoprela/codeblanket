import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedTimeSeriesModelsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'atsm-mc-1',
      question:
        'In a GARCH(1,1) model, if α = 0.08 and β = 0.90, what is the persistence and interpretation?',
      options: [
        'Persistence = 0.98, volatility shocks last very long',
        'Persistence = 0.82, volatility mean-reverts quickly',
        'Persistence = 0.08, low volatility clustering',
        'Persistence = 0.90, dominated by GARCH term',
      ],
      correctAnswer: 0,
      explanation:
        'Persistence = α + β = 0.08 + 0.90 = 0.98. High persistence (near 1) means volatility shocks decay slowly. Half-life = -log(2)/log(0.98) ≈ 35 days. Volatility mean-reverts gradually. Typical for financial markets: persistence 0.95-0.99. If persistence ≥ 1, model is non-stationary (explosive volatility).',
    },
    {
      id: 'atsm-mc-2',
      question:
        'What does a positive γ (gamma) coefficient in GJR-GARCH indicate?',
      options: [
        'Volatility is constant',
        'Leverage effect: negative returns increase volatility more than positive returns',
        'Positive returns increase volatility more',
        'No asymmetry in volatility response',
      ],
      correctAnswer: 1,
      explanation:
        'GJR-GARCH: σ²_t = ω + α*ε²_t-1 + γ*I_t-1*ε²_t-1 + β*σ²_t-1. I_t-1 = 1 if ε_t-1 < 0 (negative shock). Positive γ means negative shocks have impact (α + γ), positive shocks have impact α. Example: α = 0.05, γ = 0.10 → negative shock impact = 0.15, positive = 0.05. This is the leverage effect, common in equities.',
    },
    {
      id: 'atsm-mc-3',
      question:
        'In VAR analysis, if SPY Granger-causes TLT (p = 0.01) but TLT does NOT Granger-cause SPY (p = 0.35), what does this mean?',
      options: [
        'SPY and TLT are uncorrelated',
        "SPY predicts TLT, but TLT doesn't predict SPY (asymmetric relationship)",
        'TLT predicts SPY strongly',
        'The relationship is bidirectional',
      ],
      correctAnswer: 1,
      explanation:
        "Granger causality tests if past values of X help predict Y. SPY → TLT (p < 0.05) significant: Past SPY returns predict future TLT returns (flight-to-safety: stock selloff → bond buying). TLT → SPY (p > 0.05) not significant: Past TLT returns don't predict SPY. Asymmetric: Stocks drive bond market, not vice versa. Common in risk-on/risk-off dynamics.",
    },
    {
      id: 'atsm-mc-4',
      question: 'Why use Impulse Response Functions (IRF) in VAR models?',
      options: [
        'To forecast future returns',
        'To measure how a shock to one asset affects another asset over time',
        'To test for stationarity',
        'To calculate correlation coefficients',
      ],
      correctAnswer: 1,
      explanation:
        'IRF shows dynamic effect of 1-unit shock to asset i on asset j over h periods. Example: 1% shock to SPY → TLT response: day 0: +0.3%, day 1: +0.4%, day 3: +0.5% (peak), day 10: +0.1% (decay). Used for: Understanding shock propagation, calculating hedge ratios, assessing diversification effectiveness. Different from correlation (static) - IRF is dynamic.',
    },
    {
      id: 'atsm-mc-5',
      question: 'What is the main advantage of ARIMAX over ARIMA?',
      options: [
        'ARIMAX trains faster',
        'ARIMAX incorporates external variables (volume, volatility, sentiment) for better forecasts',
        "ARIMAX doesn't require stationarity",
        'ARIMAX has no parameters to tune',
      ],
      correctAnswer: 1,
      explanation:
        "ARIMAX = ARIMA + eXogenous variables. Y_t = ARIMA + β_1*Volume_t + β_2*VIX_t + ... Advantage: Incorporates information beyond price history. Example: Volume spike today → tomorrow's volatility prediction. VIX increase → mean reversion signal. ARIMA assumes price has all information. ARIMAX allows market microstructure variables to improve forecasts. Typically improves MAE by 10-20% when relevant variables added.",
    },
  ];
