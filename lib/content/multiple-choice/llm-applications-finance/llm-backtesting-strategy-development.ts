export const llmBacktestingStrategyDevelopmentMultipleChoice = {
  title: 'LLM-Powered Backtesting & Strategy Development - Multiple Choice',
  id: 'llm-backtesting-strategy-development-mc',
  sectionId: 'llm-backtesting-strategy-development',
  questions: [
    {
      id: 1,
      question:
        'What type of bug in LLM-generated strategy code is most dangerous because it produces plausible but incorrect results?',
      options: [
        'Syntax errors that prevent code from running',
        'Lookahead bias where future information leaks into past decisions, or off-by-one errors in lookback periods',
        'Missing comments in the code',
        'Using the wrong variable names',
      ],
      correctAnswer: 1,
      explanation:
        "Most dangerous bugs are logical errors that execute successfully but produce wrong results: lookahead bias (using tomorrow's close for today's decision), off-by-one errors (using wrong day's data), incorrect indicator calculations that are close enough to seem correct, and portfolio accounting errors. These systematically bias results without obvious failures. Syntax errors (option 0) are caught immediately, while missing comments (option 2) and wrong variable names (option 3) are less critical or cause obvious failures.",
    },
    {
      id: 2,
      question:
        'What is the primary risk of using LLMs to iterate and optimize trading strategies based on backtest results?',
      options: [
        'LLMs are too slow for strategy development',
        'Severe overfitting where the system finds parameters that fit historical noise rather than true signal, creating strategies that fail in live trading',
        'LLMs cannot generate trading code',
        'Backtesting is unnecessary with LLMs',
      ],
      correctAnswer: 1,
      explanation:
        'Primary risk is overfitting through iterative optimization. LLM suggests modifications to improve backtest, creates feedback loop where it optimizes for historical data not future performance, adds complexity that reduces generalization, and generates many variants where lucky ones are selected. Result: strategies work perfectly on training data but fail immediately when deployed. Safeguards needed: holdout data, walk-forward analysis, simplicity preference, and economic logic requirements. LLMs are fast (option 0) and can generate code (option 2), and backtesting is essential (option 3).',
    },
    {
      id: 3,
      question:
        'When should traders trust LLM explanations of why a trading strategy works?',
      options: [
        'Always trust LLM explanations because they sound plausible',
        'Never trust any LLM explanations',
        'Treat explanations as hypotheses requiring empirical validation; verify specific claims independently before accepting causal mechanisms',
        'Only trust explanations for profitable strategies',
      ],
      correctAnswer: 2,
      explanation:
        "LLMs excel at generating convincing narratives explaining any pattern, but plausibility doesn't mean correctness. Explanations might be post-hoc rationalization of random success. Proper approach: treat as hypotheses to test, validate specific claims empirically (if explanation claims week-of-month effect, verify independently), test whether explanation predicts when strategy should struggle, and maintain skepticism especially for spectacular results. \"I don't know\" often more honest than wrong story. Don't blindly trust (option 0) or reject (option 1), and profitability (option 3) doesn't validate explanation.",
    },
    {
      id: 4,
      question:
        'What is the most important validation approach beyond simple backtesting for LLM-generated strategies?',
      options: [
        'Running the backtest multiple times',
        'Walk-forward analysis with reserved holdout data, paper trading, stress testing with edge cases, and statistical analysis of trade distributions',
        'Checking that the code has no syntax errors',
        'Making sure the strategy has high returns',
      ],
      correctAnswer: 1,
      explanation:
        "Robust validation requires multiple approaches: walk-forward analysis (optimize on in-sample, test on out-of-sample), reserved holdout data never seen during development, paper trading before real capital, stress testing with edge cases (gaps, halts, extreme volatility), statistical analysis ensuring trade patterns make sense, and unit tests for components. This catches overfitting and logical errors simple backtesting misses. Multiple runs (option 0) with same data don't help, syntax checks (option 2) are basic, and high returns (option 3) might indicate overfitting.",
    },
    {
      id: 5,
      question:
        'What capability makes LLMs particularly valuable for trading strategy development compared to traditional quantitative methods?',
      options: [
        'LLMs eliminate the need for backtesting',
        'LLMs always produce better strategies',
        'LLMs can translate natural language strategy ideas into code, explain existing strategies, and iterate based on conceptual feedback while maintaining quantitative rigor',
        'LLMs guarantee profitable strategies',
      ],
      correctAnswer: 2,
      explanation:
        "LLMs bridge natural language and code: translate trader ideas into implementations, explain existing code in plain language for understanding, suggest modifications based on conceptual feedback, and generate testing frameworks. This accelerates strategy development cycle and makes quant methods accessible to non-programmers, while maintaining quantitative validation. LLMs don't eliminate backtesting (option 0), aren't always better (option 1), and can't guarantee profits (option 3). Value is productivity and accessibility, not magic.",
    },
  ],
};
