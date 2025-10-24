export const fundamentalAnalysisMLQuiz = [
  {
    id: 'faml-q-1',
    question:
      'Design a fundamental analysis ML system that predicts quarterly stock returns. Address: (1) feature selection from financial statements, (2) handling quarterly reporting lag, (3) sector normalization, (4) combining with technical signals, (5) model selection and validation. Expected accuracy and Sharpe improvement?',
    sampleAnswer:
      "Fundamental ML system: (1) Features from financial statements: Valuation: P/E, P/B, P/S, PEG, EV/EBITDA. Profitability: ROE, ROA, profit margin, operating margin. Growth: Revenue growth QoQ/YoY, earnings growth, EPS growth. Financial health: Debt/equity, current ratio, quick ratio, interest coverage. Quality: Asset turnover, receivables turnover, FCF/sales. Momentum: Earnings surprise last 4 quarters, estimate revisions. Total: ~20 features. Feature engineering: Ratios (P/E / sector_avg_P/E), changes (ROE_t - ROE_t-4), interactions (growth * quality). (2) Reporting lag: Statements released 45-60 days after quarter end. Solution: Use data as of filing date, not quarter end. Example: Q1 ends March 31, filed May 15. Use data from May 15 forward. Avoid lookahead: Don't use Q1 data on March 31. Implement 60-day lag in backtest. (3) Sector normalization: P/E=30 is high for utilities, low for tech. Normalize: z-score within sector. Feature: (stock_PE - sector_median_PE) / sector_std_PE. Sectors: Use GICS 11 sectors. Improves model 15-20% vs absolute values. (4) Combining with technical: Fundamental for selection (what to trade), Technical for timing (when to trade). Two-stage: Stage 1: Fundamental ML selects top 20% stocks (high predicted return). Stage 2: Technical signals (RSI, MACD) time entries on selected stocks. Improves Sharpe 25-40% vs fundamental-only. (5) Model: Random Forest (handles non-linearity, interactions). Parameters: n_estimators=200, max_depth=15, min_samples_leaf=10. Validation: Walk-forward quarterly. Train on 20 quarters, test on next quarter. Retrain every quarter with new data. Expected: Accuracy (direction): 55-58%, Sharpe: 1.0-1.3 (quarterly rebalancing), Information ratio: 0.5-0.8 vs S&P 500. Long-only top decile: ~5-8% annual alpha. Long-short (top vs bottom quintile): ~8-12% annual spread.",
    keyPoints: [
      'Features: 20 metrics (valuation, profitability, growth, health), engineered ratios and changes',
      'Handle 60-day reporting lag, use filing date not quarter-end',
      'Sector normalization: z-score within sector, improves 15-20%',
      'Combine: fundamental for selection, technical for timing (25-40% better Sharpe)',
      'Expect: 55-58% accuracy, Sharpe 1.0-1.3, 5-8% alpha long-only',
    ],
  },
  {
    id: 'faml-q-2',
    question:
      'Earnings surprises often correlate with price movements. Design a strategy around earnings: (1) predict earnings beats/misses, (2) position before announcement, (3) manage risk from volatility, (4) account for IV crush in options. How to extract alpha from earnings season?',
    sampleAnswer:
      'Earnings strategy: (1) Predict beats/misses: Features: Historical surprise pattern (last 8 quarters), estimate revisions (upgrades/downgrades last 30 days), whisper numbers (analyst private estimates), guidance from previous quarter, business momentum (revenue trends, new products), alternative data (web traffic, app downloads, credit card data). Model: XGBoost binary classifier (beat=1, miss=0). Training: Last 3 years, 12 quarters, 500+ stocks. Accuracy: 60-65% for direction. Confidence threshold: Only trade if predicted probability > 0.7 (eliminates low-confidence calls). (2) Positioning: Enter 2-3 days before earnings (avoid theta decay), use stock not options (IV crush risk). Position sizing: Half normal size (earnings = 2x volatility). Stop loss: 5-7% (tight, event-driven). Alternative: Buy after announcement if beat + guidance raise (momentum trade). (3) Risk management: Never position more than 2% of portfolio in single earnings play. Diversify: 10+ stocks per earnings season (uncorrelated). Use stops: If down >5% before earnings, exit (wrong setup). Hedge: If holding long, buy protective puts (cost 1-2% but limits downside to 10%). (4) IV crush in options: Problem: Options lose 30-50% value after earnings even if direction correct. Example: Stock $100, buy $105 calls for $3. Earnings: stock rises to $107 (+7%), calls now $4 (IV drops 50%). Gain only 33% instead of 200% expected. Solution: Trade stock not options for earnings. Or sell premium: Sell straddles before earnings (collect IV), buy back after (profit from crush). Risk: Unlimited loss if stock moves >10%. (5) Alpha extraction: Earnings drift: Stocks that beat often continue drifting up for 30-60 days. Strategy: Buy after announcement if beat + positive guidance, hold 30 days. Alpha: 2-3% per position, 60% win rate. Earnings season timing: High trading volume + volatility creates inefficiencies. Focus on small/mid caps (less analyst coverage, more surprises). Expected: 15-20% annual return from earnings strategy (quarterly rebalancing), Sharpe 1.2-1.5, but high risk individual trades.',
    keyPoints: [
      'Predict with XGBoost: historical surprises, estimate revisions, alternative data (60-65% accuracy)',
      'Position 2-3 days before, use stock not options (IV crush risk), half normal size',
      'Risk: 2% max per play, 10+ uncorrelated stocks, 5-7% stops',
      'Avoid options IV crush (30-50% value loss), or sell premium before earnings',
      'Earnings drift strategy: buy after beat, hold 30 days (2-3% alpha per position)',
    ],
  },
  {
    id: 'faml-q-3',
    question:
      'Alternative data (news sentiment, social media, credit card transactions) provides edge. How to: (1) collect and process each data type, (2) validate predictive power, (3) integrate with traditional fundamentals, (4) handle costs ($100-1000/month). ROI analysis?',
    sampleAnswer:
      'Alternative data strategy: (1) Collection & processing: News sentiment: NewsAPI ($50-200/month), scrape headlines every hour. NLP: FinBERT transformer (finance-specific BERT) for sentiment. Aggreg ate daily: avg_sentiment, mentions_count. Social media: Reddit API (PRAW) for r/wallstreetbets, r/stocks. Track: mention spikes (>5x baseline), sentiment (upvotes - downvotes). Twitter API ($100/month) for $TICKER mentions. Credit card data: Quandl, Second Measure ($500-1000/month). Alternative: Web scraping app download rankings, Glassdoor reviews (employee sentiment). (2) Validation: Backtest each data source independently. News sentiment → next-day return: Correlation 0.15-0.25 (weak but significant). Predictive alpha: 0.5-1% per trade. Social media spikes → 5-day return: 1-2% if spike + positive sentiment. Cost/benefit: $100/month data → need $10k+ capital to justify. Break-even: If data improves Sharpe from 1.0 to 1.15 (15%), need ~$300 monthly profit to cover $100 cost. (3) Integration with fundamentals: Ensemble approach: Fundamental score (40%), Technical score (30%), Alternative data score (30%). Alternative data as timing: Fundamental model selects stocks, alternative data times entry. Example: Stock has strong fundamentals (PE=15, ROE=25%), wait for positive sentiment spike before entering. Improves win rate 5-10%. News catalyst filter: Only trade stocks with recent positive news (removes "dead money" stocks). (4) Cost analysis: Free tier (Reddit, news scraping): $0/month, limited data. Retail tier (NewsAPI, social APIs): $100-300/month. Institutional (credit cards, satellite imagery): $1000-5000/month. ROI: Free tier: Improves Sharpe ~5% (1.0 → 1.05). Worth it even for small capital. Retail tier ($200/month): Improves Sharpe ~10-15% (1.0 → 1.15). Break-even at ~$20k capital. Institutional tier ($2000/month): Improves Sharpe ~20-30%. Need $200k+ capital. Decision rule: Alternative data spending should be < 1% of capital per year. $50k capital → max $500/year → $40/month (free-retail tier only). Expected: Alternative data adds 1-3% annual alpha, improves Sharpe 10-20%, but requires careful cost management.',
    keyPoints: [
      'Collection: NewsAPI, Reddit API, credit card data ($0-1000/month by tier)',
      'Validation: News correlation 0.15-0.25, social spikes → 1-2% returns, backtest each source',
      'Integration: 40% fundamental + 30% technical + 30% alternative, or use as timing signal',
      'Cost: <1% of capital/year, retail tier $200/month needs $20k+ capital for ROI',
      'Expected: 1-3% annual alpha, 10-20% Sharpe improvement with proper cost management',
    ],
  },
];
