import { Quiz } from '@/lib/types';

export const chartingTechnicalAnalysisMultipleChoice: Quiz = {
  title: 'Charting & Technical Analysis Tools Quiz',
  description:
    'Test your knowledge of professional charting platforms and technical analysis techniques.',
  questions: [
    {
      id: 'charting-1',
      question:
        "In TradingView\'s Pine Script v5, which function would you use to detect when a fast moving average crosses above a slow moving average?",
      options: [
        'crossover (fast_ma, slow_ma)',
        'cross (fast_ma, slow_ma)',
        'ta.crossover (fast_ma, slow_ma)',
        'ma.cross (fast_ma, slow_ma)',
      ],
      correctAnswer: 2,
      explanation:
        'In Pine Script v5, the correct syntax is `ta.crossover (source1, source2)` which returns true when source1 crosses above source2. The `ta.` prefix is required in v5 as all technical analysis functions are namespaced. This function is essential for detecting bullish crossover signals in moving average strategies.',
    },
    {
      id: 'charting-2',
      question:
        "When using Bloomberg Terminal\'s BTEC (Bloomberg Technical Analysis) function, what does a 'confidence score' associated with a detected pattern represent?",
      options: [
        'The historical profitability of trading that pattern',
        'The probability that the pattern will complete successfully',
        'The statistical reliability of the pattern detection based on how well it matches the ideal pattern geometry',
        'The number of other Bloomberg users who agree with the pattern identification',
      ],
      correctAnswer: 2,
      explanation:
        "BTEC's confidence score measures the statistical reliability of pattern detection by evaluating how closely the actual price action matches the ideal geometric characteristics of the pattern. A higher score means the detected pattern more closely resembles the textbook definition. This does NOT predict future success or probability - it only measures pattern quality. Traders should combine this score with other analysis for trading decisions.",
    },
    {
      id: 'charting-3',
      question:
        'In Volume Profile analysis, what is the primary significance of a Low Volume Node (LVN)?',
      options: [
        'It represents a price level where strong support or resistance will form',
        'It indicates a price level that price tends to move through quickly with little resistance',
        'It shows where institutional investors are most active',
        'It marks the most profitable price levels for day trading',
      ],
      correctAnswer: 1,
      explanation:
        "Low Volume Nodes (LVNs) are price levels where very little trading has occurred historically. Since few market participants have positions at these levels, there's minimal support or resistance. Price tends to move quickly through LVNs. This is in contrast to High Volume Nodes (HVNs) where significant volume creates strong support/resistance. Traders often look for quick moves through LVN zones and consolidation at HVN levels.",
    },
    {
      id: 'charting-4',
      question:
        'When implementing a MetaTrader 4 Expert Advisor (EA), what is the main difference between the `init()` and `start()` functions?',
      options: [
        '`init()` runs once when the EA is loaded; `start()` runs on every new tick',
        '`init()` handles buy orders; `start()` handles sell orders',
        '`init()` is for demo accounts; `start()` is for live accounts',
        '`init()` runs during backtesting; `start()` runs in live trading',
      ],
      correctAnswer: 0,
      explanation:
        "`init()` (or `OnInit()` in newer versions) executes once when the EA is first attached to a chart or when the chart timeframe changes. It\'s used for initialization tasks like creating indicators or validating parameters. `start()` (or `OnTick()`) executes every time a new price tick arrives, which could be multiple times per second in active markets. This is where your main trading logic resides. Efficient code structure is critical: initialize once in `init()`, check conditions and trade in `start()`.",
    },
    {
      id: 'charting-5',
      question:
        "In professional multi-timeframe analysis, which combination follows the proper 'top-down' approach for a swing trading strategy?",
      options: [
        '1-minute → 5-minute → 15-minute → 1-hour → 4-hour',
        'Monthly → Weekly → Daily → 4-hour → 1-hour',
        'Daily → 1-hour → 15-minute → 5-minute → 1-minute',
        '4-hour → Daily → Weekly → Monthly → Yearly',
      ],
      correctAnswer: 1,
      explanation:
        'The top-down approach starts with the highest timeframe to identify the overall market context and trend, then progressively moves to lower timeframes for execution. For swing trading: Monthly (market context) → Weekly (trend direction) → Daily (setup identification) → 4-hour (entry refinement) → 1-hour (execution). This prevents the common mistake of taking trades that look good on lower timeframes but are against the higher timeframe trend. Always trade in the direction of the higher timeframe bias.',
    },
  ],
};
