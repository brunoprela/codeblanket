export const bloombergTerminalMultipleChoice = [
  {
    id: '2-2-q1',
    question:
      'A hedge fund analyst needs to download 10 years of daily adjusted closing prices for the S&P 500 constituents into Excel for a backtesting analysis. Which Bloomberg Excel function is most appropriate?',
    options: [
      '=BDP("SPX INDEX", "PX_LAST") for each date',
      '=BDH("SPX INDEX", "PX_LAST", "1/1/2014", "12/31/2023")',
      '=BDS("SPX INDEX MEMBERS", "DVD_HIST_ALL")',
      '=INDIRECT("SPX"&"INDEX") with manual date entry',
      '=BQL("SPX INDEX", "historical_prices")',
    ],
    correctAnswer: 1,
    explanation:
      '=BDH() (Bloomberg Data History) is specifically designed for historical time series data. It takes a security, field name, start date, and end date, returning a two-column array of dates and values. BDP() only returns current values, BDS() is for bulk data sets (not time series), and while BQL() could work, BDH() is the standard function for this use case. The function automatically adjusts for splits and dividends if requested.',
  },
  {
    id: '2-2-q2',
    question:
      "You want to replicate Bloomberg's equity screening function (EQS) to find stocks with P/E < 15, dividend yield > 3%, and market cap > $1B. Without Bloomberg access, which Python approach is most practical?",
    options: [
      'Web scrape Bloomberg.com for the data',
      'Use yfinance to fetch data for a list of tickers and filter in pandas',
      'Purchase Bloomberg API access ($24K/year)',
      "Use Excel's built-in STOCKHISTORY function",
      'Manually search Yahoo Finance for each ticker',
    ],
    correctAnswer: 1,
    explanation:
      "Using yfinance with pandas filtering is the most practical approach. You can fetch fundamentals for a universe of tickers (like S&P 500) and filter using pandas DataFrame operations. Web scraping Bloomberg.com violates ToS, API access is expensive, STOCKHISTORY lacks fundamental data, and manual search doesn't scale. The yfinance approach: `stocks = [yf.Ticker(t).info for t in tickers]` then filter on criteria.",
  },
  {
    id: '2-2-q3',
    question:
      'A Bloomberg Terminal costs approximately $24,000 per year with a 2-year minimum contract. What is the PRIMARY reason it maintains market dominance despite this high cost?',
    options: [
      'Superior data accuracy compared to competitors',
      'The integrated ecosystem of data, news, and communication creates network effects',
      'Exclusive access to certain market data feeds',
      'Better user interface than alternatives',
      'Required by regulators for institutional trading',
    ],
    correctAnswer: 1,
    explanation:
      "The integrated ecosystem and network effects are Bloomberg's moat. Everyone in institutional finance uses it, creating a communication network (Bloomberg Messenger) that's hard to leave. You can get similar data from competitors, but you can't instantly message 325,000+ professionals. The integration of real-time data, news, analysis, and communication in one platform with shared workflows makes switching costs extremely high. It's not required by regulators or exclusively better data—it's the network.",
  },
  {
    id: '2-2-q4',
    question:
      'When using the Bloomberg Excel add-in, =BDP() formulas update in real-time during market hours, potentially causing Excel to recalculate frequently and slow down. What is the best solution for a large model with 500+ BDP formulas?',
    options: [
      'Replace all BDP() with BDH() functions',
      'Switch to manual calculation mode and use static snapshots of data',
      'Upgrade to a faster computer with more RAM',
      'Use fewer Bloomberg formulas and manually input data',
      'Set Excel calculation mode to "Automatic Except for Data Tables"',
    ],
    correctAnswer: 1,
    explanation:
      'Switching to manual calculation (Formulas → Calculation Options → Manual) and periodically refreshing to create static snapshots is the best solution. This prevents constant recalculation while preserving the formulas. You can press F9 to recalculate when needed. BDH() is for historical data (not real-time), faster computers don\'t solve the fundamental issue, manual entry defeats the purpose, and "Automatic Except Data Tables" still recalculates on every change. For production models, pull data once, copy to values, and refresh periodically.',
  },
  {
    id: '2-2-q5',
    question:
      'A quantitative researcher wants programmatic access to Bloomberg data using Python for backtesting. They have a Bloomberg Terminal on their desk. Which approach is correct?',
    options: [
      'Install `pip install bloomberg` and use the free API',
      'Use web scraping with BeautifulSoup on bloomberg.com',
      'Install `blpapi` or `pdblp` and connect to localhost:8194 (requires Terminal running)',
      'Purchase separate Bloomberg API license ($10K/year)',
      'Export data manually to CSV and read in Python',
    ],
    correctAnswer: 2,
    explanation:
      'The correct approach is installing blpapi or pdblp (pandas Bloomberg wrapper) and connecting to localhost:8194 while the Terminal is running. The Terminal must be active and logged in on the same machine. There is no "bloomberg" pip package with free API access, web scraping violates ToS, you don\'t need a separate API license if you have Terminal access, and manual CSV export doesn\'t scale for systematic research. The Bloomberg API connects to the local Terminal process.',
  },
];
