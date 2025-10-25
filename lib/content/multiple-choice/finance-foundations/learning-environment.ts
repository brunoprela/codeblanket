import { MultipleChoiceQuestion } from '@/lib/types';

export const learningEnvironmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'le-mc-1',
    question:
      'For a backtesting server running 8 hours/day, which AWS instance strategy is most cost-effective?',
    options: [
      'On-demand c6i.4xlarge ($0.68/hr = $163/month)',
      'Reserved instance c6i.4xlarge (1-year = $98/month)',
      'Spot instance c6i.4xlarge ($0.20/hr = $48/month)',
      'Lambda functions (serverless)',
    ],
    correctAnswer: 2,
    explanation:
      'Spot instances save 70%+ but can be terminated by AWS (2-minute warning). For backtesting (non-critical, can restart), spot is perfect. Cost: On-demand: $0.68 × 8 hrs × 30 days = $163/month. Spot: $0.20 × 8 × 30 = $48/month (saves $115/month). Reserved: Requires 1-year commit, saves 40% = $98/month (but paying 24hrs/day even if only using 8hrs). Lambda: Not suitable (15-minute timeout, backtests take hours). Best strategy: Spot for backtesting (interruptible okay), reserved for production (always-on), on-demand for dev (flexible). Spot risk mitigation: Checkpointing (save state every 10 min, resume if terminated), multi-AZ (try different zones if spot unavailable), fallback to on-demand if spot repeatedly fails.',
  },
  {
    id: 'le-mc-2',
    question:
      'Which Python library is best for installing technical indicators (SMA, RSI, MACD)?',
    options: [
      'pandas (built-in)',
      'TA-Lib (comprehensive)',
      'numpy (manual calculation)',
      'scikit-learn (machine learning)',
    ],
    correctAnswer: 1,
    explanation:
      "TA-Lib is industry standard for technical indicators. 150+ indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, etc. Usage: `import talib; rsi = talib.RSI(close_prices, timeperiod=14)`. Installation: Requires C library first (brew install ta-lib on macOS), then pip install TA-Lib. Alternatives: pandas-ta (pure Python, easier install but slower), pandas (can calculate manually: `df['SMA'] = df['close'].rolling(20).mean()`, tedious for complex indicators), numpy (too low-level, need to code each indicator from scratch), scikit-learn (for ML, not technical indicators). Production: TA-Lib (fastest, C implementation), Research: pandas-ta (easier), Learning: pandas manual calculation (understand the math).",
  },
  {
    id: 'le-mc-3',
    question: 'What is the purpose of a virtual environment in Python?',
    options: [
      'Run Python faster (performance boost)',
      'Isolate project dependencies (avoid conflicts)',
      'Enable GPU computing',
      'Connect to cloud services',
    ],
    correctAnswer: 1,
    explanation:
      "Virtual environments isolate dependencies per project. Problem: Project A uses pandas 1.5, Project B uses pandas 2.0 (breaking changes). Installing globally conflicts. Solution: Create venv for each project: `python -m venv projectA_env; source projectA_env/bin/activate; pip install pandas==1.5`. Benefits: No conflicts (each project isolated), Reproducibility (requirements.txt pins versions), Clean system Python (don't pollute global packages). Tools: venv (built-in, recommended), virtualenv (older, still popular), conda (manages non-Python deps too, popular for data science), pipenv (combines venv + pip), poetry (modern, handles deps + packaging). Best practice: Always use venv, Add venv/ to .gitignore (don't commit), Create requirements.txt (pip freeze > requirements.txt), Include README with setup instructions.",
  },
  {
    id: 'le-mc-4',
    question:
      'Which data format is most efficient for storing 5TB of historical OHLCV data in S3?',
    options: [
      'CSV (human-readable)',
      'JSON (flexible schema)',
      'Parquet (columnar, compressed)',
      'SQL dump (database format)',
    ],
    correctAnswer: 2,
    explanation:
      "Parquet is columnar format optimized for analytics. Comparison (5TB original CSV): CSV: 5TB, no compression, slow to query (need to scan all data). JSON: 8TB (more verbose), flexible but inefficient. Parquet: 1TB (80% compression!), columnar (query only needed columns), Snappy compression (fast), schema embedded (self-documenting). SQL dump: Not designed for S3 (for database imports). Parquet advantages: Compression (5-10× smaller), Column pruning (SELECT price: only reads price column, not date/volume), Predicate pushdown (WHERE date > '2024': Athena skips unneeded partitions), Partition support (partition by year/month for faster queries). Usage: Write: `df.to_parquet('s3://bucket/prices/year=2024/month=01/data.parquet'). Read: `pd.read_parquet('s3://bucket/prices/')`. Query: Athena SQL on S3 Parquet. Cost: 5TB CSV = $115/month, 1TB Parquet = $23/month (saves $92/month).",
  },
  {
    id: 'le-mc-5',
    question:
      'What is the main advantage of paper trading before live trading?',
    options: [
      'Paper trading has no slippage (perfect fills)',
      'Test strategy with real prices but no real money risk',
      'Paper trading is faster than live (no latency)',
      'Broker gives more capital for paper trading',
    ],
    correctAnswer: 1,
    explanation:
      'Paper trading uses REAL market prices but VIRTUAL money (\$100K default on Alpaca). Advantages: Zero risk (lose paper money, not real money), Real prices (Alpaca uses actual market data, not simulated), Test execution logic (orders, fills, cancels), Find bugs (before they cost real money), Build confidence (see strategy work in real-time). Limitations: No slippage (paper fills may be too optimistic, real fills worse), No latency (paper executions instant, real have 50-200ms delay), No psychological pressure (easy to follow plan with paper, harder with real money), Liquidity differences (paper always fills large orders, real may have partial fills). Best practice: Paper trade 3+ months (need enough trades for statistical significance), Model slippage (subtract 0.05% from fills), Gradual transition (start live with 1% capital, increase slowly), Monitor divergence (if live worse than paper, investigate why: slippage?, commission?, different fills?). When to go live: Sharpe >1.5 paper (3 months), Max drawdown <15%, Zero system errors (crashes, order failures), Comfortable with risk (can afford to lose starting capital).',
  },
];
