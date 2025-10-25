import { MultipleChoiceQuestion } from '@/lib/types';

export const readingFinancialNewsDataMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rfnd-mc-1',
      question:
        'Which SEC filing contains audited annual financial statements?',
      options: [
        '10-Q (quarterly report)',
        '10-K (annual report)',
        '8-K (current events)',
        'DEF 14A (proxy statement)',
      ],
      correctAnswer: 1,
      explanation:
        '10-K is the annual report with audited financials (CPA firm signs off). Filed within 60-90 days after fiscal year end. Contains: Full financial statements (balance sheet, income statement, cash flow, notes), MD&A (management discussion & analysis), Risk factors (Item 1A), Business description (Item 1). 10-Q = quarterly (unaudited), filed within 40-45 days. 8-K = current events (earnings, mergers, CEO changes), filed within 4 days of event. DEF 14A = proxy statement (executive comp, shareholder votes), filed before annual meeting. Form 4 = insider trading (execs buy/sell stock), filed within 2 days. For analysis: Read 10-K for deep dive (annual), 10-Q for quarterly updates, 8-K for breaking news.',
    },
    {
      id: 'rfnd-mc-2',
      question:
        'Company reports EPS of $1.50 (expected $1.40) but revenue of $50B (expected $52B). Likely market reaction?',
      options: [
        'Stock up (beat EPS)',
        'Stock down (missed revenue)',
        'Stock flat (mixed results)',
        'Depends on guidance',
      ],
      correctAnswer: 3,
      explanation:
        'Most likely: Stock DOWN despite EPS beat because revenue miss signals slowing growth. But DEPENDS on guidance: If guidance strong (expecting $55B next quarter) → stock might be up. If guidance weak or lowered → stock definitely down. Revenue > EPS because: Revenue miss often from lower sales (bad), EPS beat could be from cost-cutting (good short-term, bad long-term), Markets care more about growth (revenue) than profitability (EPS) for growth stocks. Example: Amazon often misses EPS but beats revenue → stock up because growth narrative intact. Historical pattern: EPS beat + revenue beat = +3% avg, EPS beat + revenue miss = -2% avg, EPS miss + revenue beat = -1% avg, EPS miss + revenue miss = -8% avg. Always wait for: Conference call (management explains), Guidance (forward expectations), Analyst reactions (upgrades/downgrades).',
    },
    {
      id: 'rfnd-mc-3',
      question:
        'Which Python library is best for downloading historical stock prices (free)?',
      options: [
        'pandas_datareader (deprecated)',
        'yfinance',
        'bloomberg (requires $24K/year terminal)',
        'requests (manual API calls)',
      ],
      correctAnswer: 1,
      explanation:
        'yfinance is the gold standard for free historical prices. Usage: `import yfinance as yf; data = yf.download("AAPL", start="2023-01-01")`. Pros: Free (no API key), Easy to use (one-liner), Historical data (20+ years), Multiple tickers, Fundamentals (P/E, market cap). Cons: Rate limited (too many requests = blocked temporarily), 15-min delay for real-time, Occasional gaps in data. Alternatives: pandas_datareader: Used to be standard, now mostly broken (Google/Yahoo APIs changed), Alpha Vantage: Free API key, 500 calls/day, good for intraday, Polygon: $199/month, professional quality, real-time, IEX Cloud: Pay-per-call, $0.0001-0.01 per call, Quandl: Some free datasets, mostly paid. For production: Use paid API (Polygon, IEX), For learning/backtesting: yfinance is perfect.',
    },
    {
      id: 'rfnd-mc-4',
      question: 'FRED series "DGS10" represents what data?',
      options: [
        'S&P 500 daily returns',
        '10-year Treasury yield',
        'Dollar index (DXY)',
        '10-day moving average',
      ],
      correctAnswer: 1,
      explanation:
        'DGS10 = Daily 10-year Treasury constant maturity yield (%). Why important: Risk-free rate proxy (used in CAPM, Sharpe ratio), Discount rate for DCF valuation (future cash flows), Economic indicator (low yields = recession fears, high yields = inflation/growth). Current value (2024): ~4-5%. Historical: 1980s: 10-15% (Volcker fighting inflation), 2000s: 4-6% (normal), 2008-2021: 0.5-2% (QE era), 2022-2024: 3-5% (Fed hiking). Other FRED series: DFF = Fed Funds Rate (overnight rate), UNRATE = Unemployment rate, GDP = Gross domestic product, CPIAUCSL = CPI inflation, DGS2 = 2-year Treasury (yield curve = DGS10 - DGS2), MORTGAGE30US = 30-year mortgage rate. Usage: Compare stock returns to DGS10 (is equity risk premium positive?), Bond portfolios use for duration matching, Options pricing (risk-free rate input to Black-Scholes).',
    },
    {
      id: 'rfnd-mc-5',
      question:
        'What is the latency difference between REST API polling vs WebSocket streaming for market data?',
      options: [
        'No difference (both real-time)',
        'REST: ~1-5 seconds, WebSocket: ~10-100ms',
        'WebSocket slower (more overhead)',
        'REST faster (simpler protocol)',
      ],
      correctAnswer: 1,
      explanation:
        "WebSocket dramatically faster for real-time data. REST API polling: Client polls every N seconds (e.g., GET /price/AAPL every 1 second), Latency = polling interval / 2 average = 500ms if poll every 1s, Server load increases with clients (1000 clients = 1000 requests/sec), Can't get sub-second updates efficiently. WebSocket streaming: Server pushes data when available (client doesn't poll), Latency = network latency only = 10-100ms typically, Single persistent connection (low overhead), Can get microsecond-level updates. Use cases: REST: Historical data, slow-changing data (fundamentals), simpler implementation. WebSocket: Real-time prices, order updates, high-frequency trading. Example: Alpaca WebSocket gets trades in 10-50ms, polling REST API = 500-1000ms average latency. For HFT: Co-location required (sub-millisecond), direct exchange feeds (FIX protocol), WebSocket still too slow.",
    },
  ];
