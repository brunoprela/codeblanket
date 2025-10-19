/**
 * Quiz questions for Time Series Statistics section
 */

export const timeseriesstatisticsQuiz = [
  {
    id: 'q1',
    question:
      'Why is stationarity important for time series modeling, and what happens if you ignore non-stationarity?',
    hint: 'Think about parameter stability and spurious relationships.',
    sampleAnswer:
      "**Why stationarity matters**: (1) Statistical properties (mean, variance) constant over time → reliable parameters. (2) Most models assume stationarity. (3) Forecasting requires stable relationships. **If ignored**: (1) **Spurious regression**: Non-stationary series may appear correlated when they're not. (2) Parameter instability: Estimates change with time window. (3) Invalid inference: Standard errors, tests unreliable. (4) Poor forecasts: Relationships don't hold out-of-sample. **Solution**: Test for stationarity (ADF, KPSS), difference series if needed.",
    keyPoints: [
      'Stationarity = stable statistical properties',
      'Most models assume stationarity',
      'Non-stationarity causes spurious relationships',
      'Test with ADF, KPSS',
      'Fix with differencing or detrending',
    ],
  },
  {
    id: 'q2',
    question:
      'What does "X Granger-causes Y" mean? Does this imply true causation?',
    hint: 'Consider what Granger causality actually tests.',
    sampleAnswer:
      '**Granger causality**: Past values of X help predict Y **beyond Y\'s own history**. **NOT true causation!** "Granger-cause" = "helps predict", not "causes." **Example**: Ice cream sales Granger-cause drowning. Both caused by summer (confounding). Past ice cream predicts drowning but doesn\'t cause it. **What it means**: (1) X has predictive power for Y. (2) Information in X not in Y\'s history. (3) X precedes Y temporally. **Not guaranteed**: (1) Confounders, (2) Reverse causation, (3) Coincidence. **Use**: Useful for forecasting, but be careful claiming causation.',
    keyPoints: [
      'Granger causality = predictive relationship',
      'NOT true causation',
      "X helps predict Y beyond Y's history",
      'Confounders can create Granger causality',
      'Useful for forecasting, not causal claims',
    ],
  },
  {
    id: 'q3',
    question:
      'Financial returns exhibit "volatility clustering." What does this mean, and how do you test for it?',
    hint: 'Think about autocorrelation in absolute/squared returns.',
    sampleAnswer:
      '**Volatility clustering**: Large price changes tend to be followed by large changes (either direction), and small by small. **Visually**: Periods of high volatility alternate with calm periods. **Statistically**: Returns themselves have little autocorrelation (efficient markets), but **absolute/squared returns** have positive autocorrelation. **Test**: (1) Compute absolute or squared returns. (2) Test autocorrelation (ACF, Ljung-Box test). (3) Positive autocorrelation → clustering. **Importance**: Violates IID assumption. Need GARCH/ARCH models. Risk management: volatility is predictable.',
    keyPoints: [
      'Large changes follow large changes',
      'Returns: low autocorr; |returns|: high autocorr',
      'Test with ACF of absolute/squared returns',
      'Violates IID assumption',
      'Model with GARCH family',
    ],
  },
];
