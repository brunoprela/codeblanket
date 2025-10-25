export const quizQuestions = [
    {
        id: '2-4-q1',
        question: 'A student wants to backtest a momentum trading strategy on S&P 500 stocks using 10 years of daily data. Which free data source is most suitable?',
        options: [
            'Bloomberg Terminal trial (not available)',
            'yfinance library with Python',
            'Google Finance spreadsheet function',
            'SEC EDGAR filings',
            'Manually copy data from Yahoo Finance website'
        ],
        correctAnswer: 1,
        explanation: 'yfinance is perfect for this use case: it provides free access to historical daily data back to IPO dates, handles splits/dividends automatically, supports batch downloads for multiple tickers, and integrates seamlessly with Python for backtesting. It provides adjusted close prices essential for accurate backtesting. Google Finance has limited history, SEC EDGAR doesn\'t have price data, and manual copying doesn\'t scale. yfinance can download 500+ stocks in minutes.'
    },
    {
        id: '2-4-q2',
        question: 'You need to analyze how Federal Reserve interest rate changes impact stock market returns. Which free data source provides the most comprehensive and reliable economic indicators?',
        options: [
            'Yahoo Finance economic calendar',
            'Google search for "Fed rates"',
            'FRED (Federal Reserve Economic Data) API',
            'Wikipedia economic data',
            'News articles from Bloomberg'
        ],
        correctAnswer: 2,
        explanation: 'FRED is the gold standard for economic data. It\'s maintained by the Federal Reserve Bank of St. Louis, contains 800,000+ official economic time series, provides historical data going back decades, offers a free Python API (fredapi), and is updated directly from official sources. FRED includes Fed Funds Rate, GDP, unemployment, inflation, and every major economic indicator. Yahoo Finance economic data is limited, Google/Wikipedia lack APIs and reliability, and news articles aren\'t suitable for quantitative analysis.'
    },
    {
        id: '2-4-q3',
        question: 'A quantitative researcher needs real-time (sub-second) market data for algorithmic trading. Which statement about free data sources is MOST accurate?',
        options: [
            'yfinance provides real-time data suitable for algorithmic trading',
            'Free sources provide 15-20 minute delayed data; real-time requires paid services like Polygon.io ($99-299/month)',
            'SEC EDGAR provides real-time stock prices',
            'FRED API provides real-time stock quotes',
            'All free sources now offer real-time data due to competition'
        ],
        correctAnswer: 1,
        explanation: 'Free sources like yfinance and Yahoo Finance provide 15-20 minute delayed data, which is unsuitable for algorithmic trading that requires sub-second latency. For real-time trading, you need paid services: Polygon.io ($99-299/month), IEX Cloud ($9-29/month), or Alpha Vantage premium ($49.99/month). SEC EDGAR provides company filings (not prices), FRED provides economic indicators (not stock quotes). The delay in free data is intentional - exchanges charge for real-time feeds, and free services can\'t provide it.'
    },
    {
        id: '2-4-q4',
        question: 'When using free APIs like yfinance or Alpha Vantage, what is the MOST important best practice to follow?',
        options: [
            'Make unlimited API calls to get data as fast as possible',
            'Use the data for high-frequency trading',
            'Implement rate limiting, caching, and respect API terms of service',
            'Share your API key publicly for others to use',
            'Rely solely on free sources for production trading systems'
        ],
        correctAnswer: 2,
        explanation: 'Implementing rate limiting (e.g., adding delays between requests), caching data locally (to avoid redundant API calls), and respecting API terms of service is critical. Free APIs have rate limits (Alpha Vantage: 5 calls/min, yfinance: unofficial but throttled). Exceeding limits gets your IP banned. Best practices: (1) Cache data locally, (2) Add time.sleep() between calls, (3) Never share API keys, (4) Don\'t use free sources for production HFT (they\'re delayed), (5) Have backup data sources. Respecting free services ensures they remain available for everyone.'
    },
    {
        id: '2-4-q5',
        question: 'You\'re analyzing a company\'s financials and want to verify Management Discussion & Analysis (MD&A) claims against actual financial performance. Which data source combination is best?',
        options: [
            'Bloomberg Terminal only',
            'yfinance for all data',
            'SEC EDGAR (10-K filings for MD&A) + yfinance (historical financials) + FRED (economic context)',
            'Google Finance only',
            'Wikipedia for company information'
        ],
        correctAnswer: 2,
        explanation: 'The optimal free approach combines: (1) SEC EDGAR for raw 10-K/10-Q filings containing MD&A text, risk factors, and management commentary; (2) yfinance for historical financial statements and quantitative metrics to verify MD&A claims; (3) FRED for economic indicators to understand industry context. This combination provides primary sources (SEC filings), quantitative data (yfinance), and macro context (FRED). Bloomberg costs $24K/year (overkill for this task). yfinance alone doesn\'t have full MD&A text. This demonstrates the power of combining multiple free sources for comprehensive analysis.'
    }
];

