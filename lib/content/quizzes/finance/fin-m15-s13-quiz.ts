export default {
  id: 'fin-m15-s13-quiz',
  title: 'Real-Time Risk Monitoring - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A risk system calculates full Monte Carlo VaR in 60 seconds. For pre-trade compliance checks (<100ms requirement), what is the best approach?',
      options: [
        'Buy faster computers to reduce Monte Carlo time to <100ms',
        'Skip pre-trade checks—they slow down trading',
        'Use parametric or incremental VaR for pre-trade; Monte Carlo for EOD',
        'Require traders to wait 60 seconds before each trade',
      ],
      correctAnswer: 2,
      explanation:
        'Layer multiple VaR methods by speed requirement: Pre-trade (<100ms): Parametric or incremental VaR (95% accurate but instant). Intraday (hourly): Historical VaR (recompute every hour, 98% accurate). EOD: Monte Carlo (60s, 99%+ accurate, official). Option A is impractical—even 100x faster computers only get to 0.6s. Option B eliminates critical control. Option D kills trading flow. The tradeoff is deliberate: Accept approximation for speed at real-time layer, get precision for official reporting. Compare hourly: if parametric and Monte Carlo diverge >10%, investigate. This is how large firms (Goldman, JPM) operate: Fast approximate for real-time operations, slow accurate for regulatory reporting. Both are needed—trying to use one method for all purposes fails.',
    },
    {
      id: 2,
      question:
        'Real-time P&L shows: Total $5M, Explained (delta, gamma, theta, carry) $4.8M, Unexplained $200K. What should the risk manager do?',
      options: [
        'Nothing—unexplained is only 4%',
        'Investigate immediately—unexplained should be <1%',
        'Monitor—4% is acceptable but elevated',
        'Recalculate P&L—there must be an error',
      ],
      correctAnswer: 2,
      explanation:
        'Unexplained P&L of 4% ($200K of $5M) is in the yellow zone—acceptable but elevated, warrants monitoring. <1% = green (normal rounding/timing). 1-5% = yellow (investigate but not urgent). >5% = red (investigate immediately). At 4%, should: (1) Check for common causes (missing trades? wrong prices? model approximations?), (2) Monitor through day, (3) If persists or grows, escalate. Option A is complacent—4% is not "only." Option B is overreacting—not yet red zone. Option D jumps to conclusion. Unexplained P&L is normal operating reality (complex portfolios, model approximations, timing differences), but large unexplained indicates potential issues: missing trades, pricing errors, system bugs, fraud. The 1%/5% thresholds are: <1% = model noise, >5% = something wrong. At 4%, it\'s borderline—watch closely, investigate if grows or persists multiple days.',
    },
    {
      id: 3,
      question:
        'What is the main advantage of hierarchical VaR aggregation for real-time monitoring?',
      options: [
        'It is more accurate than flat calculation',
        'Only recalculates changed sub-portfolios, not entire portfolio (100x speedup)',
        'It requires less data storage',
        'It is easier to explain to management',
      ],
      correctAnswer: 1,
      explanation:
        "Hierarchical aggregation (Firm → Desk → Trader → Position) enables incremental updates: If one trader makes a trade, only recalculate that trader's VaR, then aggregate up. Don't recalculate all 1000 traders. Speedup: Recalculating one trader (100 positions) takes 0.1s. Recalculating all traders (100K positions) takes 10s. 100x faster! Option A is wrong—accuracy is same. Option C is wrong—storage similar. Option D is irrelevant. Without hierarchy: Every trade triggers full portfolio recalc (10s latency). With hierarchy: Only affected branch recalcs (0.1s latency). This enables real-time monitoring—sub-second updates after each trade. Implementation: Maintain VaR at each level (position, trader, desk, firm), update only changed path, aggregate using correlation matrices cached at each level. Large firms (Aladdin, Bloomberg) use this architecture to monitor 10M+ positions in real-time.",
    },
    {
      id: 4,
      question:
        'A real-time risk dashboard for traders should display which metrics most prominently?',
      options: [
        'All available risk metrics for completeness',
        'P&L, VaR vs Limit, Top Positions (information traders can act on)',
        'Firm-wide metrics to show big picture',
        'Historical trends over past year',
      ],
      correctAnswer: 1,
      explanation:
        "Trader dashboards should show actionable, personal metrics prominently: (1) My P&L (am I making/losing money right now?), (2) My VaR vs Limit (how much room do I have?), (3) My top positions (what's driving my risk/P&L?). These are large, top-of-screen, color-coded. Option A creates cognitive overload—too much data paralyzes. Option C is wrong—traders focus on their book, not firm. Option D is wrong—history is secondary (tabs/drill-down). Information hierarchy principle: Most important = largest, top-left, always visible. Less important = tabs, drill-down, smaller. Traders need to glance and know: Am I OK? Can I trade more? What positions matter most? Firm-wide metrics, historical trends, detailed attribution—all useful but secondary. 80% of screen real estate should go to 20% of metrics (Pareto principle).",
    },
    {
      id: 5,
      question:
        'Why do firms use WebSocket rather than HTTP polling for real-time risk dashboards?',
      options: [
        'WebSocket is more secure',
        'WebSocket provides true push updates (server → client) with lower latency and less overhead than polling',
        'HTTP polling is deprecated technology',
        'WebSocket requires less bandwidth',
      ],
      correctAnswer: 1,
      explanation:
        'WebSocket enables bidirectional, persistent connection where server PUSHES updates to client when data changes. Contrast with HTTP polling: client asks "any updates?" every second → wasteful if no changes. WebSocket: server sends update only when VaR changes. Latency: WebSocket <100ms, polling 500-1000ms (poll interval). Overhead: WebSocket ~100 bytes/update, polling ~1KB/request (HTTP headers). Option A is wrong—both can be secure. Option C is wrong—HTTP polling still works. Option D is partially true but not the main reason. For real-time risk (P&L ticking, VaR updating, alerts firing), WebSocket is essential: 10x lower latency, 90% less bandwidth, better user experience (instant updates, not delayed by poll interval). This is why modern dashboards (Trading apps, Bloomberg Terminal, Aladdin) all use WebSocket for real-time data.',
    },
  ],
} as const;
