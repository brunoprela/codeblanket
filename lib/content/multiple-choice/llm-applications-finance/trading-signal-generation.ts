export const tradingSignalGenerationMultipleChoice = {
  title: 'Trading Signal Generation - Multiple Choice',
  id: 'trading-signal-generation-mc',
  sectionId: 'trading-signal-generation',
  questions: [
    {
      id: 1,
      question:
        'When integrating multiple signal sources (technical, fundamental, sentiment) using LLMs, what is the most appropriate way to handle conflicting signals?',
      options: [
        'Always average all signals to get a consensus',
        'Only use the signal from the most historically accurate source',
        'Explicitly identify and characterize conflicts, weight by historical accuracy in current regime, and present uncertainty rather than false confidence',
        'Randomly select one signal to follow',
      ],
      correctAnswer: 2,
      explanation:
        'Conflicts reveal uncertainty and should be made explicit, not hidden by averaging. A technical buy + fundamental sell represents different information than two neutral signals. Proper approach: identify why sources disagree (different time horizons, information sets), weight by historical accuracy in current market regime, consider signal strength not just direction, and be honest about uncertainty. LLMs can reason about conflict causes contextually. Simple averaging (option 0) loses information, single-source (option 1) ignores valuable signals.',
    },
    {
      id: 2,
      question:
        'What makes a confidence score on trading signals truly meaningful rather than misleading?',
      options: [
        'Confidence scores are always accurate if the model is large enough',
        'Higher confidence always means higher probability of profit',
        'Confidence scores must be calibrated such that stated confidence matches actual historical win rates across multiple backtests and market conditions',
        'Confidence scores should always be set to 50% to be conservative',
      ],
      correctAnswer: 2,
      explanation:
        "Meaningful confidence requires calibration: if the system says 80% confidence for 100 trades, roughly 80 should be profitable. Uncalibrated confidence creates false certainty. Calibration process: backtest signals with recorded confidence, measure actual win rate by confidence bucket, adjust scoring to match reality, and validate out-of-sample. LLMs tend toward overconfidence. Track and recalibrate regularly. Model size (option 0) doesn't guarantee calibration, high confidence doesn't guarantee profit (option 1), and fixed 50% (option 3) wastes information.",
    },
    {
      id: 3,
      question:
        'What is the primary systemic risk when many market participants use similar LLM-powered signal generation systems?',
      options: [
        'LLM API costs become prohibitive',
        'Signal crowding causes alpha degradation, correlated order flow, and potential feedback loops where LLM signals cause market moves that trigger more LLM signals',
        'Regulations will ban LLM trading',
        'LLMs will run out of training data',
      ],
      correctAnswer: 1,
      explanation:
        'When many participants use similar LLMs reading same news, trades become crowded and alpha degrades. Risks include: self-fulfilling prophecies (LLM buying creates the price move), flash crashes (LLMs all sell simultaneously), reduced market diversity, and feedback loops. This creates systemic instability. Mitigation: use proprietary data, implement contrarian signals for crowded trades, maintain diverse strategies, and monitor for crowding signs. Advantage shifts from best analysis to unique analysis. API costs (option 0) are operational not systemic.',
    },
    {
      id: 4,
      question:
        'When combining quantitative models with LLM-generated qualitative signals, what integration approach is most robust?',
      options: [
        'Replace all quantitative models with LLMs',
        'Use quantitative models exclusively and ignore LLMs',
        'Use quantitative models for primary signals and LLM analysis for signal confirmation, conflict resolution, and regime detection',
        'Randomly switch between quantitative and LLM signals',
      ],
      correctAnswer: 2,
      explanation:
        'Robust integration leverages strengths of each: quantitative models provide consistent, backtestable primary signals, while LLMs add qualitative context, confirm signals with narrative support, help resolve conflicts between quantitative signals, and detect regime changes that quantitative models might miss. This complementary approach is more reliable than using either alone. Complete replacement (option 0) loses proven quantitative rigor, ignoring LLMs (option 1) wastes valuable qualitative information.',
    },
    {
      id: 5,
      question:
        'What is the most important consideration when using LLM-generated trading signals for position sizing?',
      options: [
        'Always use maximum position size if signal is positive',
        'Use fixed position sizes regardless of signal characteristics',
        'Scale position size by signal confidence, strength, and historical accuracy, within strict risk management limits',
        'Position sizing should be random to avoid bias',
      ],
      correctAnswer: 2,
      explanation:
        "Proper position sizing considers multiple factors: signal confidence (higher confidence = larger position within limits), signal strength (strong signals warrant more capital), historical accuracy in similar situations, and always respects risk management limits (no single position should risk portfolio). This dynamic approach allocates capital efficiently while managing risk. Maximum sizing (option 0) ignores risk management, fixed sizing (option 1) doesn't optimize capital allocation, and random sizing (option 3) is nonsensical.",
    },
  ],
};
