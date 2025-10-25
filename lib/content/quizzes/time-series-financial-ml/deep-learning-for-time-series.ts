export const deepLearningForTimeSeriesQuiz = [
  {
    id: 'dlts-q-1',
    question:
      'Design a complete deep learning system for stock price forecasting. Compare LSTM, 1D CNN, and Transformer architectures: (1) When to use each model, (2) How to prevent overfitting with limited financial data (~2000 samples), (3) Feature engineering vs end-to-end learning, (4) Walk-forward validation strategy, (5) Ensemble approach. Include hyperparameter choices and expected performance ranges.',
    sampleAnswer:
      "Deep learning system design: (1) Model selection by data size: <1K samples: Don't use DL, use ARIMA/GARCH. 1K-5K samples: LSTM (simple, fewer parameters). 5K-20K samples: LSTM or 1D CNN. 20K+ samples: Transformer, complex ensembles. For SPY with 2K samples: LSTM is optimal. (2) Preventing overfitting: Use dropout (0.2-0.3), L2 regularization (weight_decay=1e-4), early stopping (patience=15-20), small architecture (hidden_size=50, layers=2), data augmentation (add noise, time warping). Validation split: 60/20/20 train/val/test temporal. Monitor train vs val loss gap. If train loss < 0.01 but val loss > 0.05 → overfitting. (3) Feature engineering: Option A - Manual features: Returns, volume, volatility, RSI, MACD → input size = 10. Pro: Domain knowledge, interpretable. Con: Missing interactions. Option B - End-to-end: Raw prices → LSTM learns representations. Pro: Automatic feature learning. Con: Needs more data, black box. Hybrid best: Engineer basic features (returns, volume) + let LSTM learn interactions. (4) Walk-forward validation: Expanding window: train on [0:t], test on [t:t+20], increment t by 20. Or rolling window: train on [t-252:t], test on [t:t+20] (last 252 days). Re-train every 20-60 days. Never use future data. Metrics: Directional accuracy (% correct direction), MAE, Sharpe of trading signals. (5) Ensemble: Train 3 models: LSTM (weight=0.5), 1D CNN (weight=0.3), Linear (weight=0.2). Average predictions. Expected improvement: 10-15% better MAE than single model. Hyperparameters: LSTM: hidden=50, layers=2, dropout=0.2, lr=0.001, batch=32, seq_len=60. CNN: filters=[32,64,128], kernel=5, dropout=0.2. Transformer: d_model=64, heads=4, layers=2. Expected performance (SPY, 1-day ahead): MAE: $1.5-3.0 (0.5-1%), Directional accuracy: 52-56%, R²: 0.05-0.15. DL rarely achieves R² > 0.2 for daily returns (markets efficient).",
    keyPoints: [
      'Model selection: <5K samples use LSTM, >20K can use Transformer',
      'Overfitting prevention: dropout 0.2-0.3, early stopping, small architecture',
      'Hybrid approach: Basic features (returns, volume) + let model learn interactions',
      'Walk-forward validation: train on past, test on future, re-train every 20-60 days',
      'Ensemble improves 10-15%, expect MAE $1.5-3 and directional accuracy 52-56%',
    ],
  },
  {
    id: 'dlts-q-2',
    question:
      'You are implementing an LSTM for multi-horizon forecasting (1, 5, 10 days ahead). Explain: (1) Direct vs recursive forecasting strategies, (2) Loss function design for multiple horizons, (3) How prediction uncertainty grows with horizon, (4) Practical applications in trading (position sizing, entry/exit timing). Compare single-output vs multi-output architectures.',
    sampleAnswer:
      "Multi-horizon forecasting: (1) Strategies: Recursive: Predict t+1, use prediction to predict t+2, etc. Pro: Single model. Con: Errors compound. Direct: Separate model for each horizon (1-day, 5-day, 10-day models). Pro: No error propagation. Con: 3× training cost. Multi-output: One model outputs [t+1, t+5, t+10]. Pro: Shared representations. Con: All horizons trained together. Best: Multi-output LSTM with separate heads. Architecture: Shared LSTM → Split into 3 heads → [pred_1day, pred_5day, pred_10day]. (2) Loss function: Option A - Weighted MSE: loss = α*MSE(1-day) + β*MSE(5-day) + γ*MSE(10-day). Weights: α=0.5, β=0.3, γ=0.2 (prioritize short-term). Option B - Multi-task learning: Each horizon has its head, separate losses, joint optimization. Include uncertainty: loss = MSE + β*std_penalty (penalize high uncertainty). (3) Uncertainty growth: Forecast at h steps: σ²(t+h) = σ²(model) + σ²(process) * h. 1-day forecast: confidence ±2%. 5-day: ±5%. 10-day: ±8%. Use Monte Carlo dropout: Make N predictions with dropout on, calculate std. High std → low confidence → reduce position size. (4) Trading applications: 1-day forecast: Intraday/daily trading. High conviction (low uncertainty), full position. 5-day forecast: Swing trading. Medium uncertainty, 50-70% position. 10-day forecast: Position sizing only, not entry. High uncertainty, adjust stops. Practical strategy: If 1-day pred up, 5-day pred up, 10-day pred up → strong signal, large position. If predictions diverge → low conviction, small position. Position sizing: size = base_size * (confidence / uncertainty). Example: 1-day pred: $500 up ± $2 → high confidence → 100% position. 5-day pred: $2500 up ± $10 → medium → 60% position. Ensemble: Average multiple models' multi-horizon predictions for robustness.",
    keyPoints: [
      'Multi-output architecture: shared LSTM, separate heads for each horizon (best approach)',
      'Weighted loss: prioritize short-term (α=0.5, β=0.3, γ=0.2 for 1/5/10 day)',
      'Uncertainty grows with horizon: ±2% (1-day), ±5% (5-day), ±8% (10-day)',
      'Trading: 1-day for entries, 5-day for swing, 10-day for sizing only',
      'Position sizing: size = base * (confidence / uncertainty), reduce with longer horizons',
    ],
  },
  {
    id: 'dlts-q-3',
    question:
      'Compare Transformers vs LSTMs for financial time series. Discuss: (1) Attention mechanism advantages for market data, (2) Parameter efficiency and training time, (3) Interpretability of attention weights, (4) When Transformers outperform LSTMs, (5) Practical limitations with small financial datasets. Include code architecture patterns.',
    sampleAnswer:
      'Transformers vs LSTMs: (1) Attention advantages: LSTM processes sequentially (t-59 → t-58 → ... → t-1 → t). Transformer uses attention: all time steps attend to each other simultaneously. Benefit: Captures long-range dependencies (t-59 can directly influence t without sequential processing). Example: Fed announcement at t-40 directly attends to current price at t. LSTM must propagate through 40 steps. Market use case: Earnings date at t-30 → current prediction at t (attention directly connects). (2) Parameters: LSTM(hidden=50, layers=2): ~50K parameters. Transformer (d_model=64, heads=4, layers=2): ~150K parameters. Training time: LSTM: Sequential, cannot parallelize time dimension. Transformer: Parallel, 2-3× faster training. But: More parameters → needs more data. (3) Interpretability: Extract attention weights: attention_matrix[60x60] shows which past days influence current prediction. High attention at t-1, t-5, t-21 (1 day, 1 week, 1 month lags) → model learned technical patterns. Visualization: Heatmap of attention weights → understand what model focuses on. LSTM: Hidden states not interpretable. (4) When Transformers win: Large datasets (>10K samples): Transformer learns complex patterns. Multiple related series: Multi-variate Transformer (stocks + bonds + macro). Long sequences (seq_len > 100): Attention handles long-range better than LSTM. When LSTM wins: Small datasets (<5K): LSTM has fewer parameters, less overfitting. Simple patterns: LSTM sufficient. Real-time inference: LSTM faster (smaller model). (5) Practical limitations: Financial data often <10K samples → Transformers overfit. Solution: Pre-training on related assets (transfer learning). Train on SPY+QQQ+IWM (3× data), fine-tune on target. Use smaller Transformer: d_model=32, heads=2 instead of 64, 4. Regularization: dropout=0.3, weight_decay=1e-3. Architecture: class FinancialTransformer: input[60 days] → Linear projection[1→64] → Positional encoding → TransformerEncoder (d_model=64, heads=4, layers=2) → Linear[64→1]. Expect: 10-20% better than LSTM only with >10K samples.',
    keyPoints: [
      'Attention captures long-range dependencies directly (Fed announcement → current price)',
      'Transformer 2-3× faster training (parallel), but 3× more parameters (needs more data)',
      'Attention weights interpretable: heatmap shows which past days influence prediction',
      'Transformers win with >10K samples, LSTMs win with <5K (fewer parameters)',
      'Practical: Pre-train on multiple assets, use smaller architecture (d_model=32, heads=2)',
    ],
  },
];
